import torch
import torch.nn as nn
import torch.nn.functional as F

from gnnfree.utils.utils import SmartTimer
from gnnfree.nn.models.basic_models import MLPLayers
import dgl.function as fn

def edge_msg_func(edges):
            
    msg = F.relu(edges.src['h'] + edges.data['e'])

    return {'msg': msg}


class GINELayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats):
        super(GINELayer, self).__init__()
        self.mlp = MLPLayers(2, [in_feats, 2*in_feats, out_feats])
        self.edge_mlp = MLPLayers(2, [edge_feats, 2*edge_feats, in_feats])
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, g, node_feat, edge_feat):
        with g.local_scope():
            g.ndata['h'] = node_feat
            g.edata['e'] = self.edge_mlp(edge_feat)
            g.update_all(edge_msg_func, fn.sum(msg='msg', out='out_h'))
            # g.update_all(fn.copy_u('h','msg'), fn.sum(msg='msg', out='out_h'))
            out = self.mlp((1+self.eps)*node_feat+g.ndata['out_h'])
            return out

class GINLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GINLayer, self).__init__()
        self.mlp = MLPLayers(2, [in_feats, 2*in_feats, out_feats])
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(fn.copy_u('h','msg'), fn.sum(msg='msg', out='out_h'))
            out = self.mlp((1+self.eps)*feature+g.ndata['out_h'])
            return out
