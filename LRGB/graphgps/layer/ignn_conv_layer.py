# NOTE: This code is a reimplemented version of IGNN for ease of understanding and use.
# Please also cite IGNN <https://arxiv.org/abs/2009.06211> if using our code.

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


PROPS = 300
TOL = 1e-3


def norm_inf(weights, startfrom=0, kappa=0.99):
    # make abssum <= kappa for each col by minusing something
    w = weights.clone().detach().cpu()
    wabs = w.abs()
    cl1 = wabs.sum(dim=0) - kappa
    idx = torch.where(cl1 > 0)[0]
    idx = idx[idx >= startfrom]
    for i, l1 in zip(idx, cl1[idx]):
        cabs = wabs[:, i]
        for j, x in enumerate(cabs.sort().values):
            minus = l1 / (wabs.shape[0] - j)
            if minus <= x:
                break
            l1 -= x
        w[:, i] = w[:, i].sign() * F.relu(cabs - minus)
    weights.copy_(w)
    return weights


def norm_adj(edges, n, asym=False, add_self=1, return_spectral_rad=False):
    es = edges[:, edges[0] != edges[1]]
    dev = es.device
    deg = torch.zeros(n).to(dev)
    # Add Self-Loops
    if add_self:
        es = torch.cat((
            torch.arange(n).to(dev).view(1, -1).repeat(2, 1), es), dim=1)
    else:
        deg += 1e-5
    ew = torch.ones(es.shape[1]).to(dev)
    if add_self:
        ew[:n] = add_self
    deg.scatter_add_(dim=0, index=es[0], src=ew)
    if asym:
        val = ew * (deg ** -1)[es[0]]
    else:
        val = ew * (deg ** -0.5)[es].prod(dim=0)
    adj = torch.sparse_coo_tensor(es, val, (n, n))
    if not return_spectral_rad:
        return adj
    coo = sp.coo_matrix(
        (np.abs(val.cpu().numpy()), es.cpu().numpy()),
        shape=adj.shape)
    spec_rad = np.abs(sp.linalg.eigs(
        coo, k=1, return_eigenvectors=False)[0]) + 1e-5
    return adj, spec_rad


class IGNN(nn.Module):
    def __init__(self, gnn, hid, dout, dropout=0):
        super(self.__class__, self).__init__()

        self.gnn = gnn

        self.edges = None
        self.adj = None
        self.adj_rho = None

        self.igc = IGConv(hid, F.relu)
        self.classifier = nn.Sequential(
            # nn.LayerNorm(hid),
            nn.Dropout(dropout),
            nn.Linear(hid, dout),
        )

    def forward(self, x, edges):
        if edges is not self.edges:
            self.edges = edges
            self.adj, self.spec_rad = norm_adj(
                edges, x.shape[0], return_spectral_rad=True)

        x = self.gnn(x, edges)
        x = self.igc(self.adj, x, self.spec_rad)
        x = F.normalize(x, dim=1)
        x = self.classifier(x)
        return x


class IGConv(nn.Module):
    def __init__(self, feats, activator, kappa=0.9):
        super(IGConv, self).__init__()
        self.kappa = kappa
        self.w = nn.Parameter(torch.randn(feats, feats) * (feats ** -0.5))
        # act should be element-wise
        self.act = activator
        self.z = None
        self.zshape = ()

    def forward(self, adj, bias, spec_rad=1.0):
        bshape = tuple(bias.shape)
        if self.zshape != bshape:
            self.z = torch.zeros(bshape).to(bias.device)
            self.zshape = bshape
        norm_inf(self.w.data, kappa=self.kappa/spec_rad)
        return IFunc.apply(self.act, adj, self.z, self.w, bias)


class IFunc(Function):
    @staticmethod
    def loop(act, adj, z, w, bias, iters=PROPS, tol=TOL):
        with torch.no_grad():
            tol = tol * z.abs().max().item()
            err = 0
            for _ in range(iters):
                z0, z = z, act(adj @ (z @ w) + bias)
                err = (z0 - z).norm(float('inf'))
                if err < tol:
                    break
            return z

    @staticmethod
    def forward(ctx, act, adj, z0, w, bias):
        z = IFunc.loop(act, adj, z0, w, bias, iters=PROPS, tol=TOL)
        with torch.enable_grad():
            h = adj @ (z @ w) + bias
            h.requires_grad_(True)
            z = act(h)
            dact = torch.autograd.grad(z.sum(), h)[0]
        ctx.save_for_backward(w, z, adj, dact, z0)
        return z

    @staticmethod
    def backward(ctx, dz):
        # NOTE: if H = XW, then dX = dH W^T, dW = X^T dH
        w, z, adj, dact, z0 = ctx.saved_tensors
        # for H = AZW, dZ = A^T dAZ = A^T dH W^T
        dh0 = IFunc.loop(
            lambda dz0: dz0 * dact, adj.T, z0, w.T, dz, iters=PROPS, tol=TOL)
        # dW = (AZ)^T dH
        dw = (adj @ z).T @ dh0
        return None, None, None, dw, dh0
