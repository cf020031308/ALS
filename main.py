import os
import time
import json
import datetime
import argparse
import psutil

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Function as ADFunc
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric import datasets as pyg_data
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import f1_score, roc_auc_score
import torch_sparse


TOL = 1e-3
parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, default='MLP', help=(
    'MLP | SGC | LPA | PPNP | PPNR | GIN | GCN | SAGE | GCNII'
))
parser.add_argument('dataset', type=str, default='cora', help=(
    'cora | citeseer | pubmed | flickr | arxiv | yelp | reddit | ...'
))
parser.add_argument('--runs', type=int, default=1, help='Default: 1')
parser.add_argument('--gpu', type=int, default=0, help='Default: 0')
parser.add_argument(
    '--split', type=float, default=0,
    help=('Ratio of labels for training.'
          ' Set to 0 to use default split (if any) or 0.6. '
          ' With an integer x the dataset is splitted like Cora with the '
          ' training set be composed by x samples per class. '
          ' Default: 0'))
parser.add_argument(
    '--lr', type=float, default=0.001, help='Learning Rate. Default: 0.001')
parser.add_argument(
    '--dropout', type=float, default=0.0, help='Default: 0')
parser.add_argument('--n-layers', type=int, default=2, help='Default: 2')
parser.add_argument(
    '--weight-decay', type=float, default=0.0, help='Default: 0')
parser.add_argument(
    '--early-stop-epochs', type=int, default=200,
    help='Maximum epochs until stop when accuracy decreasing. Default: 200')
parser.add_argument(
    '--max-epochs', type=int, default=2000,
    help='Maximum epochs. Default: 2000')
parser.add_argument(
    '--hidden', type=int, default=32,
    help='Dimension of hidden representations and implicit state. Default: 32')
parser.add_argument(
    '--heads', type=int, default=1,
    help='Heads for GAT. Default: 1')
parser.add_argument(
    '--alpha', type=float, default=0.1,
    help='Hyperparameter for GCNII/LPAs. Default: 0.1')
parser.add_argument(
    '--beta', type=float, default=0.0,
    help='Hyperparameter for GCNII/AGLS. Default: 0.05')
parser.add_argument(
    '--props', type=int, default=300,
    help='number of propagations for convergence of LPA/PPNP/LaE/...')
parser.add_argument(
    '--tol', type=float, default=TOL,
    help='tolerance for convergence of LPA/PPNP/LaE/...')
parser.add_argument(
    '--no-self-loop', action='store_true',
    help='DO NOT add self-loops')
parser.add_argument(
    '--weighted', action='store_true',
    help='use weighted edges in LPAs and PPNPs')
parser.add_argument(
    '--implicit', action='store_true',
    help='use implicit differentiation in LPAs and PPNPs')
parser.add_argument(
    '--with-cg', action='store_true',
    help='Solving linear systems with CG')
parser.add_argument(
    '--with-stop', action='store_true',
    help='Skip solved linear systems in CG')
parser.add_argument(
    '--with-basis', action='store_true',
    help='Start CG with the known eigenvector')
parser.add_argument(
    '--adj-norm', type=str, default='sym',
    help='Way to normalize the adjacency matrix (sym|row|col). Default: sym')
parser.add_argument(
    '--asym', action='store_true',
    help='asym')
parser.add_argument(
    '--batched', action='store_true',
    help='batched')
parser.add_argument(
    '--frame', type=str, default='',
    help='version of implementation')
parser.add_argument(
    '--dot-attention', action='store_true',
    help='Transformer-style Attentions (instead of GATv2-style)')
parser.add_argument(
    '--skip-connections', type=int, default=0,
    help='Number of skip connections. Default: 0')
args = parser.parse_args()

inf = float('inf')
if not torch.cuda.is_available():
    args.gpu = -1
print(datetime.datetime.now(), args)
script_time = time.time()

g_dev = None
gpu = lambda x: x
if args.gpu >= 0:
    g_dev = torch.device('cuda:%d' % args.gpu)
    gpu = lambda x: x.to(g_dev)
coo = torch.sparse_coo_tensor


class Optim(object):
    def __init__(self, params):
        self.params = params
        self.opt = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay)

    def __repr__(self):
        return 'params: %d' % sum(p.numel() for p in self.params)

    def __enter__(self):
        self.opt.zero_grad()
        self.elapsed = time.time()
        return self.opt

    def __exit__(self, *vs, **kvs):
        self.opt.step()
        self.elapsed = time.time() - self.elapsed


class GCNII(nn.Module):
    def __init__(self, din, hidden, n_layers, dout, dropout=0, **kw):
        super(self.__class__, self).__init__()
        self.lin1 = nn.Linear(din, hidden)
        self.convs = nn.ModuleList([
            gnn.GCN2Conv(
                channels=hidden,
                alpha=kw['alpha'],
                theta=kw['beta'],
                layer=i + 1,
            ) for i in range(n_layers)])
        self.lin2 = nn.Linear(hidden, dout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x0 = x = F.relu(self.lin1(self.dropout(x)))
        for conv in self.convs:
            x = F.relu(conv(self.dropout(x), x0, edge_index))
        return self.lin2(self.dropout(x))


class DPP(ADFunc):
    @staticmethod
    def spppr(adj, x, alpha, iters=300, tol=TOL, batch_size=0, *args, **kwargs):
        adj = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(adj)
        z = x * 0
        for _ in range(iters):
            z0, z = z, (1 - alpha) * (adj @ z) + alpha * x
            if tol and ((z0 - z).norm(inf) < tol):
                break
        return z

    @staticmethod
    def ppr(adj, x, alpha, iters=300, tol=TOL, fix=(), *args, **kwargs):
        z = x * 0
        for _ in range(iters):
            z0, z = z, (1 - alpha) * torch.sparse.mm(adj, z) + alpha * x
            if fix:
                z[fix[0]] = fix[1]
            if tol and ((z0 - z).norm(inf) < tol):
                break
        return z

    @staticmethod
    def forward(ctx, adj, y, alpha, n_layers, tol, batch_size):
        with torch.no_grad():
            z = DPP.ppr(adj, y, alpha, iters=n_layers, tol=tol)
        ctx.save_for_backward(
            adj, torch.tensor([
                alpha, n_layers, tol, batch_size]).to(y.device), z)
        return z

    @staticmethod
    def backward(ctx, dz):
        adj, (alpha, iters, tol, bs), z = ctx.saved_tensors
        with torch.no_grad():
            dz = DPP.ppr(adj.T, dz / alpha, alpha, int(iters), tol)
        da = None
        if adj.requires_grad:
            s, t = e = adj._indices()
            bs = int(bs)
            if bs:
                dw = torch.empty(e.shape[1]).to(z.device)
                for idx in DataLoader(range(e.shape[1]), batch_size=bs):
                    dw[idx] = (
                        dz[s[idx]].unsqueeze(1) @ z[t[idx]].unsqueeze(-1)
                    ).squeeze(-1).squeeze(-1)
            else:
                dw = (
                    dz[s].unsqueeze(1) @ z[t].unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
            da = (1 - alpha) * coo(e, dw, (z.shape[0], z.shape[0]))
        return da, alpha * dz, None, None, None, None


def load_data(name):
    is_bidir = None
    train_masks = None
    W = None
    if args.dataset in (
        'roman_empire', 'amazon_ratings', 'minesweeper',
        'tolokers', 'questions',
    ):
        data = numpy.load('dataset/hetgs/%s.npz' % args.dataset)
        X, Y, E, train_masks, valid_masks, test_masks = map(data.get, [
            'node_features', 'node_labels', 'edges',
            'train_masks', 'val_masks', 'test_masks'])
        X, Y, E, train_masks, valid_masks, test_masks = map(torch.from_numpy, [
            X, Y, E.T, train_masks, valid_masks, test_masks])
        is_bidir = False
    elif args.dataset in ('arxiv', 'mag', 'products', 'proteins'):
        ds = NodePropPredDataset(name='ogbn-%s' % args.dataset)
        train_idx, valid_idx, test_idx = map(
            ds.get_idx_split().get, 'train valid test'.split())
        if args.dataset == 'mag':
            train_idx = train_idx['paper']
            valid_idx = valid_idx['paper']
            test_idx = test_idx['paper']
        g, labels = ds[0]
        if args.dataset == 'mag':
            labels = labels['paper']
            g['edge_index'] = g['edge_index_dict'][('paper', 'cites', 'paper')]
            g['node_feat'] = g['node_feat_dict']['paper']
        E = torch.from_numpy(g['edge_index'])
        if args.dataset == 'proteins':
            W = torch.from_numpy(g['edge_feat'])
            X = torch.zeros(g['num_nodes'], W.shape[1]).scatter_add_(
                dim=0,index=E[0].view(-1, 1).expand_as(W), src=W)
        else:
            W = None
            X = torch.from_numpy(g['node_feat'])
        Y = torch.from_numpy(labels).squeeze(-1)
        n_nodes = X.shape[0]
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True
        is_bidir = False
        train_masks = [train_mask] * args.runs
        valid_masks = [valid_mask] * args.runs
        test_masks = [test_mask] * args.runs
    else:
        dn = 'dataset/' + args.dataset
        g = (
            pyg_data.Planetoid(dn, name='Cora') if args.dataset == 'cora'
            else pyg_data.Planetoid(dn, name='CiteSeer') if args.dataset == 'citeseer'
            else pyg_data.Planetoid(dn, name='PubMed') if args.dataset == 'pubmed'
            else pyg_data.CitationFull(dn, name='Cora') if args.dataset == 'corafull'
            else pyg_data.CitationFull(dn, name='Cora_ML') if args.dataset == 'coraml'
            else pyg_data.CitationFull(dn, name='DBLP') if args.dataset == 'dblp'
            else pyg_data.Reddit(dn) if args.dataset == 'reddit'
            else pyg_data.Reddit2(dn) if args.dataset == 'reddit-sp'
            else pyg_data.Flickr(dn) if args.dataset == 'flickr'
            else pyg_data.Yelp(dn) if args.dataset == 'yelp'
            else pyg_data.AmazonProducts(dn) if args.dataset == 'amazon'
            else pyg_data.WebKB(dn, args.dataset.capitalize())
            if args.dataset in ('cornell', 'texas', 'wisconsin')
            else pyg_data.WikipediaNetwork(dn, args.dataset)
            if args.dataset in ('chameleon', 'crocodile', 'squirrel')
            else pyg_data.WikiCS(dn) if args.dataset == 'wikics'
            else pyg_data.Actor(dn) if args.dataset == 'actor'
            else pyg_data.Coauthor(dn, name='CS') if args.dataset == 'coauthor-cs'
            else pyg_data.Coauthor(dn, name='Physics') if args.dataset == 'coauthor-phy'
            else pyg_data.Amazon(dn, name='Computers') if args.dataset == 'amazon-com'
            else pyg_data.Amazon(dn, name='Photo') if args.dataset == 'amazon-photo'
            else None
        ).data
        X, Y, E, train_mask, valid_mask, test_mask = map(
            g.get, 'x y edge_index train_mask val_mask test_mask'.split())
        if args.dataset in (
                'amazon-com', 'amazon-photo', 'coauthor-cs', 'coauthor-phy'):
            train_mask, valid_mask, test_mask = torch.zeros(
                (3, X.shape[0]), dtype=bool)
            train_idx, valid_idx, test_idx = map(
                numpy.load('dataset/split/%s.npz' % args.dataset).get,
                ['train', 'valid', 'test'])
            train_mask[train_idx] = True
            valid_mask[valid_idx] = True
            test_mask[test_idx] = True
        elif args.dataset in (
            'cora', 'citeseer', 'pubmed', 'corafull',
            'reddit', 'flickr', 'yelp', 'amazon'
        ):
            train_masks = [train_mask] * args.runs
            valid_masks = [valid_mask] * args.runs
            test_masks = [test_mask] * args.runs
            is_bidir = True
        elif args.dataset in ('wikics', ):
            train_masks = train_mask.T
            valid_masks = valid_mask.T
            test_masks = [test_mask] * args.runs
        else:
            train_masks = [train_mask[:, i % train_mask.shape[1]]
                           for i in range(args.runs)]
            valid_masks = [valid_mask[:, i % valid_mask.shape[1]]
                           for i in range(args.runs)]
            test_masks = [test_mask[:, i % test_mask.shape[1]]
                          for i in range(args.runs)]
            is_bidir = False
    if is_bidir is None:
        for i in range(E.shape[1]):
            src, dst = E[:, i]
            if src.item() != dst.item():
                print(src, dst)
                break
        is_bidir = ((E[0] == dst) & (E[1] == src)).any().item()
        print('guess is bidir:', is_bidir)
    n_labels = int(Y.max().item() + 1)
    is_multilabel = len(Y.shape) == 2
    # Save Label Transitional Matrices
    fn = 'dataset/labeltrans/%s.json' % args.dataset
    if not (is_multilabel or os.path.exists(fn)):
        with open(fn, 'w') as file:
            mesh = coo(
                Y[E], torch.ones(E.shape[1]), size=(n_labels, n_labels)
            ).to_dense()
            den = mesh.sum(dim=1, keepdim=True)
            mesh /= den
            mesh[den.squeeze(1) == 0] = 0
            json.dump(mesh.tolist(), file)
    # Remove Self-Loops
    E = E[:, E[0] != E[1]]
    # Get Undirectional Edges
    if not is_bidir:
        E = torch.cat((E, E[[1, 0]]), dim=1)
    if (train_masks is None or train_masks[0] is None) and not args.split:
        args.split = 0.6
    nrange = torch.arange(X.shape[0])
    if 0 < args.split < 1:
        torch.manual_seed(42)  # the answer
        train_masks, valid_masks, test_masks = [], [], []
        for _ in range(args.runs):
            train_mask = torch.zeros(X.shape[0], dtype=bool)
            valid_mask = torch.zeros(X.shape[0], dtype=bool)
            test_mask = torch.zeros(X.shape[0], dtype=bool)
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
            if is_multilabel:
                val_num = test_num = int((1 - args.split) / 2 * X.shape[0])
                idx = torch.randperm(X.shape[0])
                train_mask[idx[val_num + test_num:]] = True
                valid_mask[idx[:val_num]] = True
                test_mask[idx[val_num:val_num + test_num]] = True
            else:
                for c in range(n_labels):
                    label_idx = nrange[Y == c]
                    val_num = test_num = int(
                        (1 - args.split) / 2 * label_idx.shape[0])
                    perm = label_idx[torch.randperm(label_idx.shape[0])]
                    train_mask[perm[val_num + test_num:]] = True
                    valid_mask[perm[:val_num]] = True
                    test_mask[perm[val_num:val_num + test_num]] = True
    elif int(args.split):
        # NOTE: work only for graphs with single labelled nodes.
        torch.manual_seed(42)  # the answer
        train_masks, valid_masks, test_masks = [], [], []
        for _ in range(args.runs):
            train_mask = torch.zeros(X.shape[0], dtype=bool)
            for y in range(n_labels):
                label_mask = Y == y
                train_mask[
                    nrange[label_mask][
                        torch.randperm(label_mask.sum())[:int(args.split)]]
                ] = True
            valid_mask = ~train_mask
            valid_mask[
                nrange[valid_mask][torch.randperm(valid_mask.sum())[500:]]
            ] = False
            test_mask = ~(train_mask | valid_mask)
            test_mask[
                nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]
            ] = False
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
    return X, Y, E, W, train_masks, valid_masks, test_masks, is_bidir


class Stat(object):
    def __init__(self):
        self.preprocess_time = 0
        self.training_times = []
        self.evaluation_times = []

        self.best_test_scores = []
        self.best_times = []
        self.best_training_times = []

        self.mem = psutil.Process().memory_info().rss / 1024 / 1024
        self.gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            self.gpu = torch.cuda.memory_allocated(g_dev) / 1024 / 1024

    def start_preprocessing(self):
        self.preprocess_time = time.time()

    def stop_preprocessing(self):
        self.preprocess_time = time.time() - self.preprocess_time

    def start_run(self):
        self.params = None
        self.scores = []
        self.acc_training_times = []
        self.acc_times = []
        self.training_times.append(0.)
        self.evaluation_times.append(0.)

    def record_training(self, elapsed):
        self.training_times[-1] += elapsed

    def record_evaluation(self, elapsed):
        self.evaluation_times[-1] += elapsed

    def evaluate_result(self, y):
        self.scores.append([
            get_score(Y[m], y[m])
            for m in [train_mask, valid_mask, test_mask]])
        self.acc_training_times.append(self.training_times[-1])
        self.acc_times.append(self.preprocess_time + self.training_times[-1])
        self.best_epoch = torch.tensor(self.scores).max(dim=0).indices[1] + 1
        dec_epochs = len(self.scores) - self.best_epoch
        if dec_epochs == 0:
            self.best_acc = self.scores[-1][1]
            self.best_y = y
        return dec_epochs >= args.early_stop_epochs

    def end_run(self):
        if self.scores:
            self.scores = torch.tensor(self.scores)
            print('train scores:', self.scores[:, 0].tolist())
            print('val scores:', self.scores[:, 1].tolist())
            print('test scores:', self.scores[:, 2].tolist())
            print('acc training times:', self.acc_training_times)
            print('max scores:', self.scores.max(dim=0).values)
            idx = self.scores.max(dim=0).indices[1]
            self.best_test_scores.append((idx, self.scores[idx, 2]))
            self.best_training_times.append(self.acc_training_times[idx])
            self.best_times.append(self.acc_times[idx])
            print('best test score:', self.best_test_scores[-1])

    def end_all(self):
        conv = 1.0 + torch.tensor([
            idx for idx, _ in self.best_test_scores])
        score = 100 * torch.tensor([
            score for _, score in self.best_test_scores])
        tm = torch.tensor(self.best_times)
        ttm = torch.tensor(self.best_training_times)
        print('converge time: %.3f±%.3f' % (
            tm.mean().item(), tm.std().item()))
        print('converge training time: %.3f±%.3f' % (
            ttm.mean().item(), ttm.std().item()))
        print('converge epochs: %.3f±%.3f' % (
            conv.mean().item(), conv.std().item()))
        print('score: %.2f±%.2f' % (score.mean().item(), score.std().item()))

        # Output Used Time
        print('preprocessing time: %.3f' % self.preprocess_time)
        for name, times in (
            ('total training', self.training_times),
            ('total evaluation', self.evaluation_times),
        ):
            times = torch.tensor(times or [0], dtype=float)
            print('%s time: %.3f±%.3f' % (
                name, times.mean().item(), times.std().item()))

        # Output Used Space
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu = torch.cuda.max_memory_allocated(g_dev) / 1024 / 1024
        print('pre_memory: %.2fM + %.2fM = %.2fM' % (
            self.mem, self.gpu, self.mem + self.gpu))
        print('max_memory: %.2fM + %.2fM = %.2fM' % (
            mem, gpu, mem + gpu))
        print('memory_diff: %.2fM + %.2fM = %.2fM' % (
            mem - self.mem,
            gpu - self.gpu,
            mem + gpu - self.mem - self.gpu))


X, Y, E, W, train_masks, valid_masks, test_masks, is_bidir = load_data(
    args.dataset)
n_nodes = X.shape[0]
n_features = X.shape[1]
is_multilabel = len(Y.shape) == 2
n_labels = Y.shape[1] if is_multilabel else int(Y.max().item() + 1)
deg = E.shape[1] / n_nodes
print('nodes: %d' % n_nodes)
print('features: %d' % n_features)
print('classes: %d' % n_labels)
print('is_multilabel:', is_multilabel)
print('edges without self-loops: %d' % (E.shape[1] / 2))
print('average degree: %.2f' % deg)
train_sum = sum([m.sum() for m in train_masks]) / len(train_masks)
valid_sum = sum([m.sum() for m in valid_masks]) / len(valid_masks)
test_sum = sum([m.sum() for m in test_masks]) / len(test_masks)
print('split: %d (%.2f%%) / %d (%.2f%%) / %d (%.2f%%)' % (
    train_sum, 100 * train_sum / n_nodes,
    valid_sum, 100 * valid_sum / n_nodes,
    test_sum, 100 * test_sum / n_nodes,
))
eh = (
    (Y[E[0]] == Y[E[1]]).sum().float()
    / E.shape[1] / (n_labels if is_multilabel else 1))
print('intra_rate: %.2f%%' % (100 * eh))

ds = torch.zeros(n_nodes)
ds.scatter_add_(dim=0, index=E[0], src=torch.ones(E.shape[1]))
if is_multilabel:
    ds = ds.unsqueeze(-1).repeat(1, n_labels)
    hs = torch.zeros(n_nodes, n_labels)
    for i in range(n_labels):
        hs[:, i].scatter_add_(
            dim=0, index=E[0], src=(Y[E[0], i] == Y[E[1], i]).float())
else:
    hs = torch.zeros(n_nodes)
    hs.scatter_add_(dim=0, index=E[0], src=(Y[E[0]] == Y[E[1]]).float())
nh = (hs / ds)[ds > 0].mean()
print('node homophily: %.2f%%' % (100 * nh))

if not is_multilabel:
    d2 = sum([ds[Y == i].sum() ** 2 for i in range(n_labels)])
    d2 *= E.shape[1] ** -2
    ah = (eh - d2) / (1 - d2)
    print('adjusted homophily: %.2f%%' % (100 * ah))

if is_multilabel:
    _cri = nn.BCEWithLogitsLoss(reduction='none')
    criterion = lambda x, y: _cri(x, y.float()).sum(dim=1)
    sg = torch.sigmoid
    if args.dataset in ('proteins', ):
        get_score = lambda y_true, y_pred: roc_auc_score(
            y_true.cpu(), y_pred.cpu()).item()
    else:
        get_score = lambda y_true, y_pred: f1_score(
            y_true.cpu(), (y_pred > 0.5).cpu(), average='micro').item()
else:
    criterion = lambda x, y: F.cross_entropy(x, y, reduction='none')
    sg = lambda x: torch.softmax(x, dim=-1)
    if args.dataset in ('minesweeper', 'tolokers', 'questions', ):
        get_score = lambda y_true, y_pred: roc_auc_score(
            y_true.cpu(), (1 - y_pred[:, 0]).cpu()).item()
    else:
        get_score = lambda y_true, y_pred: f1_score(
            y_true.cpu(), y_pred.argmax(dim=-1).cpu(), average='micro').item()


def norm_adj(edges, n, norm='sym'):
    deg = torch.zeros(n).to(edges.device)
    deg.scatter_add_(
        dim=0, index=edges[0],
        src=torch.ones(edges.shape[1]).to(edges.device))
    # with open('degree_counts/%s_train.txt' % args.dataset, 'w') as file:
    #     for xs in deg.unique(sorted=True, return_counts=True):
    #         file.write(','.join('%d' % x for x in xs))
    #         file.write('\n')
    if norm == 'row':
        val = (deg ** -1)[edges[0]]
    elif norm == 'col':
        val = (deg ** -1)[edges[1]]
    else:
        val = (deg ** -0.5)[edges].prod(dim=0)
    return coo(edges, val, (n, n))


if not args.no_self_loop:
    E = torch.cat((
        torch.arange(n_nodes).view(1, -1).repeat(2, 1), E), dim=1)
ev = Stat()
opt = None

# Preprocessing
ev.start_preprocessing()

X, Y = map(gpu, [X, Y])
E = gpu(E)
A = None

ev.stop_preprocessing()

for run in range(args.runs):
    train_mask = train_masks[run]
    valid_mask = valid_masks[run]
    test_mask = test_masks[run]
    if is_multilabel:
        train_y = Y[train_mask].float()
    else:
        train_y = F.one_hot(Y[train_mask], n_labels).float()
        
    torch.manual_seed(run)
    torch.cuda.manual_seed_all(run)
    ev.start_run()

    if args.method == 'MLP':
        net = gpu(gnn.MLP(
            [n_features, *([args.hidden] * (args.n_layers - 1)), n_labels],
            dropout=args.dropout))
        opt = Optim([*net.parameters()])
        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                z = net(X)
                criterion(z[train_mask], Y[train_mask]).mean().backward()
            ev.record_training(opt.elapsed)
            if ev.evaluate_result(sg(z)):
                break
    elif args.method in ('SGC', 'LPA', 'PPNP', 'PPNR'):
        '''
        LPA/PPNP/PPNR: propagate labels/predictions/representations
        args.weighted: weighted edges combined with GATv2 and MAGNA
        args.implicit: implicit differentiation borrowed from IGNN
        '''
        params = []
        batch_size = int(1 + E.shape[1] / args.hidden)
        # construct adjacency matrix w or w/o edge weights
        if args.weighted:
            down = gpu(nn.Linear(n_features, args.hidden, bias=False))
            kern = gpu(nn.Bilinear(args.hidden, args.hidden, 1))
            params.extend([
                p for f in (down, kern) for p in f.parameters()])

            def get_A(x, e):
                h = down(x)
                if batch_size:
                    w = torch.empty(e.shape[1]).to(x.device)
                    for idx in DataLoader(
                            range(e.shape[1]), batch_size=batch_size):
                        w[idx] = kern(*h[e[:, idx]]).squeeze(-1)
                else:
                    w = kern(*h[e]).squeeze(-1)
                return torch.sparse.softmax(
                    coo(e, w, (n_nodes, n_nodes)), dim=1)
        else:
            if A is None:
                A = norm_adj(E, n_nodes, norm=args.adj_norm)
            get_A = lambda x, e: A
        # propagate features/labels/predictions/representations
        if args.method == 'SGC':
            get_h = lambda x, e: x
            pred = gpu(nn.Sequential(
                nn.Linear(n_features, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, n_labels),
                nn.Softmax(dim=1)))
            params.extend([*pred.parameters()])
        elif args.method == 'LPA':
            ty = torch.ones(n_nodes, n_labels).to(train_y.device)
            ty[train_mask] = train_y
            get_h = lambda x, e: ty
            pred = lambda x: x
        elif args.method == 'PPNP':
            enc = gpu(nn.Sequential(
                nn.Linear(n_features, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, n_labels),
                nn.Softmax(dim=1)))
            params.extend([*enc.parameters()])
            get_h = lambda x, e: enc(x)
            pred = lambda x: x
        else:
            enc = gpu(nn.Sequential(
                nn.Linear(n_features, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.ReLU()))
            get_h = lambda x, e: F.relu(enc(x))
            pred = gpu(nn.Sequential(
                nn.Linear(args.hidden, n_labels),
                nn.Softmax(dim=1)))
            params.extend([*enc.parameters(), *pred.parameters()])
        # optimize w or w/o implicit differentiation
        net = lambda: pred((DPP.apply if args.implicit else DPP.spppr)(
            get_A(X, E), get_h(X, E),
            args.alpha, args.n_layers, args.tol, batch_size))
        # train and evaluate
        if params:
            opt = Optim(params)
            for epoch in range(1, 1 + args.max_epochs):
                with opt:
                    y = net()
                    hy = y[train_mask, Y[train_mask]]
                    hy.backward(hy - 1)
                ev.record_training(opt.elapsed)
                if ev.evaluate_result(y):
                    break
        else:
            t = time.time()
            y = net()
            ev.record_training(time.time() - t)
            ev.evaluate_result(y)
    else:
        # Label as Equilibrium <https://arxiv.org/abs/2103.13355>
        if args.method == 'PG':
            from als import PPRGAT
            net = PPRGAT(n_features, n_labels, **args.__dict__)
        elif args.method == 'MPG':
            from als import MultiPPRGAT
            net = MultiPPRGAT(n_features, n_labels, **args.__dict__)
        elif args.method == 'GAT':
            net = gnn.GAT(
                n_features, args.hidden, args.n_layers, n_labels,
                v2=True, dropout=args.dropout)
        elif args.method == 'IGCN':
            from ignn import IGNN
            if args.n_layers > 1:
                backbone = gnn.GCN(
                    n_features, args.hidden, args.n_layers - 1,
                    args.hidden, args.dropout, bias=False)
                net = IGNN(backbone, args.hidden, n_labels)
            elif args.n_layers == 1:
                backbone = gnn.MLP(
                    [n_features, args.hidden],
                    dropout=args.dropout, bias=False)
                fwd = backbone.forward
                backbone.forward = lambda x, E: fwd(x)
                net = IGNN(backbone, args.hidden, n_labels)
            else:
                net = IGNN(lambda x, e: x, n_features, n_labels)
        else:
            # from models import GCN
            net = {
                'GIN': gnn.GIN,
                'GCN': gnn.GCN,
                'SAGE': gnn.GraphSAGE,
                'GCNII': GCNII,
            }[args.method](
                n_features, args.hidden, args.n_layers, n_labels, args.dropout)
        net = gpu(net)
        opt = Optim([*net.parameters()])
        if run == 0:
            print('params: 0' if opt is None else opt)

        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                z = net(X, E)
                criterion(z[train_mask], Y[train_mask]).mean().backward()
            ev.record_training(opt.elapsed)

            # Inference
            t = time.time()
            with torch.no_grad():
                net.eval()
                state = sg(net(X, E))
                net.train()
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(state):
                break
            # print('epoch:', epoch, 'score:', ev.scores[-1])

    ev.end_run()
ev.end_all()
print('script time:', time.time() - script_time)
