import torch
import torch_geometric.nn as pyg_nn
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcn_conv_layer import GCNConvLayer
from graphgps.layer.als_conv_layer import MAPR
from graphgps.layer.ignn_conv_layer import IGNN
from graphgps.layer.eignn_conv_layer import EIGNN
from graphgps.layer.gpr_conv_layer import GPR_prop


class ALSConvLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual):
        super(self.__class__, self).__init__()
        self.residual = residual
        self.act = torch.nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            torch.nn.Dropout(dropout))
        self.model = MAPR(
            dim_in, dim_out, edge_dim=dim_in,
            implicit=True, weighted=True,
            with_cg=True, with_stop=True, with_basis=True,
            asym=cfg.gnn.als.asym,
            dot_attention=cfg.gnn.als.dot_attention,
            heads=cfg.gnn.als.heads,
            alpha=cfg.gnn.als.alpha,
            skip_connections=cfg.gnn.als.skip_connections)

    def forward(self, batch):
        x_in, batch.x = batch.x, self.act(self.model(
            batch.x, batch.edge_index, edge_attr=batch.edge_attr))
        if self.residual:
            batch.x = x_in + batch.x
        return batch


class IConvLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual):
        super(self.__class__, self).__init__()
        self.residual = residual
        self.act = torch.nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            torch.nn.Dropout(dropout))
        gcn1 = pyg_nn.GCNConv(dim_in, dim_in, bias=False)
        self.model = IGNN(gcn1, dim_in, dim_out)

    def forward(self, batch):
        x_in, batch.x = batch.x, self.act(self.model(
            batch.x, batch.edge_index))
        if self.residual:
            batch.x = x_in + batch.x
        return batch


class EIConvLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual):
        super(self.__class__, self).__init__()
        self.residual = residual
        self.act = torch.nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            torch.nn.Dropout(dropout))
        self.model = EIGNN(dim_in, dim_out)

    def forward(self, batch):
        x_in, batch.x = batch.x, self.act(self.model(batch))
        if self.residual:
            batch.x = x_in + batch.x
        return batch


class APPNPLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual):
        super(self.__class__, self).__init__()
        self.residual = residual
        self.act = torch.nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            torch.nn.Dropout(dropout))
        self.enc = pyg_nn.MLP([dim_in, dim_in, dim_out])
        self.model = pyg_nn.APPNP(K=10, alpha=cfg.gnn.ppr)

    def forward(self, batch):
        x_in, batch.x = batch.x, self.act(self.model(
            self.enc(batch.x), batch.edge_index))
        if self.residual:
            batch.x = x_in + batch.x
        return batch


class GPRLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual):
        super(self.__class__, self).__init__()
        self.residual = residual
        self.act = torch.nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            torch.nn.Dropout(dropout))
        self.enc = pyg_nn.MLP([dim_in, dim_in, dim_out])
        self.model = GPR_prop(K=10, alpha=cfg.gnn.ppr)

    def forward(self, batch):
        x_in, batch.x = batch.x, self.act(self.model(
            self.enc(batch.x), batch.edge_index))
        if self.residual:
            batch.x = x_in + batch.x
        return batch


@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'gcnconv':
            return GCNConvLayer
        elif model_type == 'als':
            return ALSConvLayer
        elif model_type == 'ignn':
            return IConvLayer
        elif model_type == 'eignn':
            return EIConvLayer
        elif model_type == 'appnp':
            return APPNPLayer
        elif model_type == 'gprgnn':
            return GPRLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
