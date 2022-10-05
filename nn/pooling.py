import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl

from torch_scatter import scatter

from gnnfree.nn.models.basic_models import MLPLayers


class Pooler(nn.Module):
    def get_out_dim(self):
        pass

class GDPool(Pooler):
    def __init__(self, emb_dim, gd_deg=True) -> None:
        super().__init__()
        self.gd_deg = gd_deg
        self.emb_dim = emb_dim
        if gd_deg:
            self.mlp_combine_gd_deg = MLPLayers(2, h_units=[emb_dim+1, 2*emb_dim, emb_dim], batch_norm=False)
        self.mlp_combine_nei_gd = MLPLayers(2, h_units=[2*emb_dim+1, 4*emb_dim, emb_dim])
        self.mlp_combine_node_nei = MLPLayers(2, h_units=[2*emb_dim, 4*emb_dim, emb_dim])

    def get_out_dim(self):
        return self.emb_dim

    def forward(self, repr, nodes, neighbors, neighbor_count, dist, gd, gd_count, gd_deg):

        neighbors_repr = repr[neighbors]
        gd_repr = repr[gd]
        if self.gd_deg:
            combined_gd_repr = self.mlp_combine_gd_deg(torch.cat([gd_repr, gd_deg.view(-1,1)], dim=-1))
        else:
            combined_gd_repr = gd_repr
        combined_gd_repr = scatter(combined_gd_repr, torch.arange(len(gd_count), device=repr.device).repeat_interleave(gd_count), dim=0, dim_size=len(gd_count))
        combined_repr = self.mlp_combine_nei_gd(torch.cat([combined_gd_repr, neighbors_repr, dist.view(-1,1)], dim=-1))
        combined_repr = scatter(combined_repr, torch.arange(len(neighbor_count), device=repr.device).repeat_interleave(neighbor_count), dim=0, dim_size=len(neighbor_count))

        node_repr = self.mlp_combine_node_nei(torch.cat([combined_repr, repr[nodes]], dim=-1))
        return node_repr


class LinkReprExtractor(Pooler):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.emb_dim = emb_dim
    
    def get_out_dim(self):
        return self.emb_dim

    def forward(repr):
        pass 

class NodePairExtractor(LinkReprExtractor):
    def get_out_dim(self):
        return self.emb_dim*2

    def forward(self, repr, head, tail):
        return torch.cat([repr[head], repr[tail]], dim=-1)

class RelExtractor(LinkReprExtractor):
    def __init__(self, emb_dim, num_rels) -> None:
        super().__init__(emb_dim)
        self.rel_emb = nn.Embedding(num_rels, emb_dim, sparse=False)

    def forward(self, rel):
        return self.rel_emb(rel)

class HorGDExtractor(LinkReprExtractor):
    def forward(self, repr, geodesic, gd_len):
        gd_repr = repr[geodesic]
        gd_repr = scatter(gd_repr, torch.arange(len(gd_len), device=repr.device).repeat_interleave(gd_len), dim=0, dim_size=len(gd_len))
        return gd_repr

class VerGDExtractor(LinkReprExtractor):
    def __init__(self, emb_dim, gd_deg=False) -> None:
        super().__init__(emb_dim)
        self.gd_deg = gd_deg
        if gd_deg:
            self.mlp_combine_gd_deg = MLPLayers(2, h_units=[emb_dim+1, 2*emb_dim, emb_dim])
        self.mlp_head_gd_process = MLPLayers(2, h_units=[emb_dim, 2*emb_dim, emb_dim])
        self.mlp_tail_gd_process = MLPLayers(2, h_units=[emb_dim, 2*emb_dim, emb_dim])

    def get_out_dim(self):
        return self.emb_dim

    def get_ver_gd_one_side(self, repr, gd, gd_len, gd_deg):
        gd_repr = repr[gd]
        # if len(gd_repr)==0:
        #     return torch.zeros((len(gd_len), self.params.emb_dim), device=repr.device)
        if gd_deg:
            gd_repr = self.mlp_combine_gd_deg(torch.cat([gd_repr, gd_deg.view(-1,1)],dim=-1))
        gd_repr = scatter(gd_repr, torch.arange(len(gd_len), device=repr.device).repeat_interleave(gd_len), dim=0, dim_size=len(gd_len))
        return gd_repr


    def forward(self, repr, head_gd, tail_gd, head_gd_len, tail_gd_len, head_gd_deg=None, tail_gd_deg=None):
        head_gd_repr = self.get_ver_gd_one_side(repr, head_gd, head_gd_len, head_gd_deg)
        tail_gd_repr = self.get_ver_gd_one_side(repr, tail_gd, tail_gd_len, tail_gd_deg)
        return self.mlp_head_gd_process(head_gd_repr)+self.mlp_tail_gd_process(tail_gd_repr)

class DistExtractor(Pooler):
    def get_out_dim(self):
        return 1
    
    def forward(self, dist):
        return dist.view(-1,1)

feature_module_dict = {'node_pair':NodePairExtractor, 'rel':RelExtractor, 'HorGD':HorGDExtractor, 'VerGD': VerGDExtractor, 'dist': DistExtractor, '':None}