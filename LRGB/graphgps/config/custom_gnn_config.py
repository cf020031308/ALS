from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """
    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.heads = 4
    cfg.gnn.attn_dropout = 0.1

    cfg.gnn.use_vn = True
    cfg.gnn.vn_pooling = 'add'

    cfg.gnn.norm_type = 'layer'

    # ALS configuration
    cfg.gnn.als = CN()
    cfg.gnn.als.asym = False
    cfg.gnn.als.dot_attention = False
    cfg.gnn.als.heads = 0
    cfg.gnn.als.alpha = 0.0
    cfg.gnn.als.skip_connections = 0

    # PPR alpha for PPNP and GPRGNN
    cfg.gnn.ppr = 0.5
