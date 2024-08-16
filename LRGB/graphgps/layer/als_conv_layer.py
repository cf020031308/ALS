import typing

import math
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
try:
    from gatv2_conv import GATv2Conv
except Exception:
    from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter
import torch_sparse
import torch_geometric.nn as gnn


diff = lambda x1, x2: (x1 - x2).norm(float('inf')).item()
dot = lambda x1, x2: (x1 * x2).sum(dim=-1)
ITERS, TOL = 300, 1e-3


def ec(e, x1, x2, c=dot, bs=0, *size):
    if bs:
        m = e.shape[1]
        r = torch.empty(m, *size).to(x1.device)
        for i in range(0, m, bs):
            end = i + bs
            r[i:end] = c(x1[e[0, i:end]], x2[e[1, i:end]])
    else:
        r = c(x1[e[0]], x2[e[1]])
    return r


class PPR(Function):
    @staticmethod
    def mm(ew, z, bs=0, T=False, **kwargs):
        # Batched Edge Computing
        if isinstance(ew, torch_sparse.SparseTensor):
            return (ew @ z.squeeze(1)).unsqueeze(1)
        r, c = T + 0, 1 - T
        e, w = ew
        if bs:
            ret = w.new_zeros(z.shape)
            for i in range(0, w.shape[0], bs):
                end = i + bs
                src = w[i:end].unsqueeze(-1) * z[e[c, i:end]]
                index = e[r, i:end].view(
                    -1, *([1] * (src.dim() - 1))).expand_as(src)
                ret.scatter_add_(dim=0, index=index, src=src)
            return ret
        return scatter(
            w.unsqueeze(-1) * z[e[c]], e[r],
            dim=0, dim_size=z.shape[0], reduce='sum')

    @staticmethod
    def power(ew, x, gamma, iters, tol, batch_size, basis, with_stop, **kwargs):
        if basis is None:
            z = x * 0
        else:
            basis = F.normalize(basis, p=2, dim=0).unsqueeze(-1)
            z = gamma * basis * (basis * x).sum(dim=0, keepdim=True) + (1 - gamma) * x
        ret = minz = z.clone()
        for i in range(iters):
            z0 = z
            minz[:] = z = gamma * PPR.mm(ew, z, batch_size, **kwargs) + (1 - gamma) * x
            if tol:
                rm = (z - z0).norm(float('inf'), dim=0) < tol
                if rm.all().item():
                    break
                if with_stop:
                    # Eliminate heads and channels
                    rh = rm.all(dim=1)
                    rc = rm.all(dim=0)
                    drh = rh.any().item()
                    if drh or rc.any().item():
                        hm = torch.arange(rh.shape[0]).to(rh.device)[~rh]
                        if drh and not isinstance(
                                ew, torch_sparse.SparseTensor):
                            ew = (ew[0], ew[1][:, hm])
                        hm = hm.view(-1, 1)
                        cm = ~rc
                        x = x[:, hm, cm]
                        z = z[:, hm, cm]
                        gamma = gamma[:, hm, cm]
                        minz = minz[:, hm, cm]
        return ret

    @staticmethod
    def cg(ew, x, gamma, iters, tol, batch_size, basis, with_stop, **kwargs):
        A = lambda z: (
            z - gamma * PPR.mm(ew, z, batch_size, **kwargs)) / (1 - gamma)
        # Initialize with the normalized basis
        if basis is None:
            z = x * 0
            r = x
        else:
            basis = F.normalize(basis, p=2, dim=0).unsqueeze(-1)
            z = basis * (basis * x).sum(dim=0, keepdim=True)
            r = x - z
        p = r.clone()
        ret = minz = z.clone()
        rr = (r ** 2).sum(dim=0, keepdim=True)
        minr = 1 / rr.new_zeros(rr.shape[1:])

        for i in range(iters):
            ap = A(p)
            alpha = (rr + 1e-16) / ((p * ap).sum(dim=0, keepdim=True) + 1e-16)
            z = z + alpha * p
            r = r - alpha * ap
            rn = r.norm(p=float('inf'), dim=0)
            rm = rn <= minr
            if rm.any().item():
                minr[rm] = rn[rm]
                minz[:, rm] = z[:, rm]
                if tol:
                    rm = rn < tol
                    if rm.all().item():
                        break
                    if with_stop:
                        # Eliminate heads and channels
                        rh = rm.all(dim=1)
                        rc = rm.all(dim=0)
                        drh = rh.any().item()
                        if drh or rc.any().item():
                            hm = torch.arange(rh.shape[0]).to(rh.device)[~rh]
                            if drh and not isinstance(
                                    ew, torch_sparse.SparseTensor):
                                ew = (ew[0], ew[1][:, hm])
                            hm = hm.view(-1, 1)
                            cm = ~rc
                            r = r[:, hm, cm]
                            rr = rr[:, hm, cm]
                            minr = minr[hm, cm]
                            p = p[:, hm, cm]
                            z = z[:, hm, cm]
                            minz = minz[:, hm, cm]
                            gamma = gamma[:, hm, cm]
            rr0, rr = rr, (r ** 2).sum(dim=0, keepdim=True)
            p = r + (rr + 1e-16) / (rr0 + 1e-16) * p
        return ret

    solver = power

    @staticmethod
    def ppr(
            e, w, x, gamma, iters, tol, batch_size,
            basis=None, with_stop=False, T=False):
        assert not x.isnan().any().item()
        assert not w.isnan().any().item()
        # rescale x for robustness
        s = x.detach().abs().max().item()
        if not s:
            return x
        x = x / s
        if torch_sparse.cuda_spec and batch_size == 0 and w.shape[1] == 1:
            # NOTE: spmm is faster than scatter
            r, c = T + 0, 1 - T
            n = x.shape[0]
            ew = torch_sparse.SparseTensor(
                row=e[r], col=e[c], value=w.squeeze(1), sparse_sizes=(n, n))
        else:
            ew = (e, w)
        z = PPR.solver(
            ew, x, gamma, iters, tol, batch_size,
            basis=basis, with_stop=with_stop, T=T)
        assert not z.isnan().any().item(), w.abs().max().item()
        return z * s

    @staticmethod
    @torch.no_grad()
    def forward(ctx, e, w, x, gamma, iters, tol, batch_size, basis, with_stop):
        z = PPR.ppr(
            e, w, x, gamma,
            iters=iters, tol=tol, batch_size=batch_size,
            basis=basis, with_stop=with_stop, T=False)
        confs = torch.tensor([
            iters, tol, batch_size, with_stop]).to(x.device)
        ctx.save_for_backward(e, w, gamma, confs, x, z, basis)
        return z

    @staticmethod
    @torch.no_grad()
    def backward(ctx, dz):
        e, w, gamma, (iters, tol, bs, stop), x, z, basis = ctx.saved_tensors
        assert not dz.isnan().any().item(), z.abs().max().item()
        dz = PPR.ppr(
            e, w, dz / (1 - gamma), gamma,
            iters=int(iters), tol=float(tol), batch_size=int(bs),
            basis=basis, with_stop=bool(stop), T=True)
        dw = ec(e, gamma * dz, z, dot, int(bs), *z.shape[1:-1])
        assert not dw.isnan().any().item(), z.abs().max().item()
        da = dz * (z - (1 - gamma) * x) / gamma
        # da = None
        return None, dw, (1 - gamma) * dz, da, None, None, None, None, None


class MAPR(nn.Module):
    r"""
    Args:
        batched (bool, optional): Compute edge features in batches.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        implicit: bool = False,
        props: int = ITERS,
        tol: float = TOL,
        alpha: float = 0.1,
        bias: bool = True,
        batched: bool = False,
        asym: bool = False,
        weighted: bool = False,
        skip_connections: int = 1,
        with_cg: bool = False,
        with_stop: bool = False,
        with_basis: bool = False,
        dot_attention: bool = False,
        negative_slope: float = 0.2,
        edge_dim: typing.Optional[int] = None,
        post_norm: bool = True,
        **kwargs,
    ):
        super(self.__class__, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        H = self.heads = heads if weighted else 1
        C = out_channels // H
        assert H * C == out_channels
        self.implicit = implicit
        self.alpha = alpha
        self.bias = bias
        self.batched = batched
        self.asym = asym
        self.weighted = weighted
        self.with_cg = with_cg and alpha < 0.35 and implicit and not asym
        self.with_basis = with_basis
        self.negative_slope = negative_slope
        self.ppr = lambda e, w, x, gamma, bs, basis: (
            PPR.apply if implicit else PPR.ppr)(
                e, w, x, gamma, props, tol, bs, basis, with_stop)
        init_w = lambda *args: nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(*args)))
        init_b = lambda *args: nn.Parameter(torch.zeros(*args)) if bias else 0
        self.lin_w = init_w(in_channels, out_channels)
        self.lin_b = init_b(1, H, C)
        if weighted:
            self.att_w = init_w(in_channels, out_channels)
            self.att_b = init_b(1, H, C)
            self.e_enc = lambda x: x
            if edge_dim:
                self.lin_e = init_w(edge_dim, out_channels)
                self.e_enc = lambda x: (x @ self.lin_e).view(-1, H, C)
            if dot_attention:
                self.att = lambda h1, h2, ex=1: (
                    (h1 + self.att_b) * (h2 + self.att_b) * self.e_enc(ex)
                ).sum(dim=-1) / (C ** 0.5)
            else:
                self.att_a = init_w(1, H, C)
                self.att = lambda h1, h2, ex=0: (self.att_a * F.leaky_relu(
                    h1 + h2 + self.att_b + self.e_enc(ex), negative_slope
                )).sum(dim=-1)
        self.skip_connections = nn.ParameterList([
            init_w(in_channels, out_channels)
            for _ in range(skip_connections)])
        self.ln = nn.LayerNorm(out_channels) if post_norm else nn.Identity()
        if alpha == 0:
            self.gamma = nn.Parameter(torch.randn(1, H, C))
        else:
            self.gamma = torch.ones(1, H, C) * (1 - alpha)

    def forward(self, x, e, edge_attr=None):
        n, m, H = x.shape[0], e.shape[1], self.heads
        C = self.out_channels // H
        edge_batch = n if self.batched else 0
        PPR.solver = PPR.cg if self.with_cg else PPR.power
        # Graph Attentions
        v = (x @ self.lin_w).view(n, H, C)
        if self.weighted:
            q = (x @ self.att_w).view(n, H, C)
            k = v if self.asym else q
            if edge_attr is None:
                w = ec(e, q, k, self.att, edge_batch, H)
            elif edge_batch:
                w = torch.empty(m, H).to(v.device)
                for i in range(0, m, edge_batch):
                    end = i + edge_batch
                    w[i:end] = self.att(
                        q[e[0, i:end]], k[e[1, i:end]], edge_attr[i:end])
            else:
                w = self.att(q[e[0]], k[e[1]], edge_attr)
            w_max = scatter(w.detach(), e[0], dim=0, dim_size=n, reduce='max')
        else:
            w = torch.zeros(m, H).to(v.device)
            w_max = torch.zeros(n, H).to(v.device)
        basis = None
        if self.asym:
            if self.with_basis:
                basis = torch.ones(n, H).to(v.device)
            w = (w - w_max[e[0]]).exp()
            d = scatter(w, e[0], dim=0, dim_size=n, reduce='sum')
            w = w / (d[e[0]] + 1e-16)
        else:
            if self.with_basis:
                basis = scatter(
                    (w - w.max(dim=0).values.unsqueeze(0)).detach().exp(),
                    e[0], dim=0, dim_size=n, reduce='sum') ** 0.5
            d = scatter(
                (w - w_max[e[0]]).exp(),
                e[0], dim=0, dim_size=n, reduce='sum') ** 0.5
            w = (
                w - (w_max[e[0]] + w_max[e[1]]) / 2
            ).exp() / (d[e[0]] * d[e[1]] + 1e-16)
        # Long-range message passing
        if self.alpha == 0:
            gamma = 0.01 + 0.98 * torch.sigmoid(self.gamma)
        else:
            gamma = self.gamma.to(v.device)
        v = self.ppr(e, w, v, gamma, edge_batch, basis)
        # Short-range message passing
        for sk in self.skip_connections:
            v = (
                gamma * PPR.mm((e, w), v, bs=edge_batch)
                + (1 - gamma) * (x @ sk).view(n, H, C))
        if self.bias:
            v = v + self.lin_b
        return self.ln(v.view(n, -1))


class PPRGAT(nn.Module):
    def __init__(
            self, in_channels, out_channels, hidden, n_layers,
            heads, dropout, frame='', edge_dim=None, **kwargs):
        super(self.__class__, self).__init__()
        hidden_channels = hidden * heads
        self.enc = gnn.MLP(
            [in_channels] + [hidden_channels] * n_layers, act='gelu',
            norm='LayerNorm', dropout=dropout, plain_last=False
        ) if n_layers else nn.Identity()
        self.e_enc = gnn.MLP(
            [edge_dim] + [hidden_channels] * n_layers, act='gelu',
            norm='LayerNorm', dropout=dropout, plain_last=False
        ) if (n_layers and edge_dim) else nn.Identity()
        self.appr = {
            'gat': GATv2Conv, 'pyg': MAPR_PyG
        }.get(frame.lower(), MAPR)(
            hidden_channels if n_layers else in_channels,
            hidden_channels, heads,
            edge_dim=hidden_channels if n_layers and edge_dim else edge_dim,
            **kwargs)
        self.dec = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels))

    def forward(self, x, e, edge_attr=None, **kwargs):
        x = self.enc(x)
        if edge_attr is not None:
            edge_attr = self.e_enc(edge_attr)
        x = self.appr(x, e, edge_attr)
        x = self.dec(x)
        return x


class MultiPPRGAT(nn.Module):
    def __init__(
            self, in_channels, out_channels, hidden, n_layers,
            heads, dropout, frame='', edge_dim=None, **kwargs):
        super(self.__class__, self).__init__()
        hidden_channels = hidden * heads
        appr = {'gat': GATv2Conv, 'pyg': MAPR_PyG}.get(frame.lower(), MAPR)
        self.enc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout),
            nn.GELU())
        self.e_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.Dropout(dropout),
            nn.GELU()) if edge_dim else nn.Identity()
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.convs = nn.ModuleList([
            appr(hidden_channels, hidden_channels, heads,
                 edge_dim=hidden_channels if edge_dim else None, **kwargs)
            for _ in range(n_layers)])
        self.fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout))
            for _ in range(n_layers)])
        self.pred = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, out_channels))

    def forward(self, x, e, edge_attr=None, **kwargs):
        x = self.enc(x)
        if edge_attr is not None:
            edge_attr = self.e_enc(edge_attr)
        for norm, conv, fn in zip(self.norms, self.convs, self.fns):
            x = x + fn(conv(norm(x), e, edge_attr))
        x = self.pred(x)
        return x
