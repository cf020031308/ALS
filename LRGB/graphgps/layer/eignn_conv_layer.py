import torch
import torch.nn as nn
import scipy.linalg
import scipy.sparse as sp

from .EIGNN.normalization import aug_normalized_adjacency
from .EIGNN.utils import sparse_mx_to_torch_sparse_tensor
from .EIGNN.functions import IDMFunction


class EIGNN(nn.Module):
    'modified from https://github.com/liu-jc/EIGNN/tree//6a2c8e73c11bfebc8614d955226dbae600cc8dfc'
    def __init__(self, din, dout, gamma=0.8):
        super(self.__class__, self).__init__()
        self.F = nn.Parameter(torch.FloatTensor(din, din))
        self._gamma = torch.tensor(gamma, dtype=torch.float)
        self.gamma = nn.Parameter(self._gamma)
        self.B = nn.Linear(din, dout, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.F)
        self.gamma.data = self._gamma
        self.B.reset_parameters()

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        if batch.get('adj') is None:
            adj = sp.coo_matrix(
                (torch.ones(edge_index.shape[1]), edge_index.cpu()),
                shape=(x.shape[0], x.shape[0]))
            sp_adj = aug_normalized_adjacency(adj)
            batch.adj = sparse_mx_to_torch_sparse_tensor(sp_adj, device=x.device)
            sy = (abs(sp_adj - sp_adj.T) > 1e-10).nnz == 0
            Lambda_S, Q_S = (scipy.linalg.eigh if sy else scipy.linalg.eig)(sp_adj.toarray())
            batch.Lambda_S = torch.from_numpy(Lambda_S).type(torch.FloatTensor).view(-1, 1).to(x.device)
            batch.Q_S = torch.from_numpy(Q_S).type(torch.FloatTensor).to(x.device)

        return self.B(IDMFunction.apply(
            x.T, self.F, batch.adj, batch.Q_S, batch.Lambda_S, self.gamma).T)
