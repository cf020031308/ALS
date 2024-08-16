import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcn_conv_layer import GCNConvLayer
from graphgps.layer.als_conv_layer import MAPR


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
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
