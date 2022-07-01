import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import RelGraphConv

from gnnfree.nn.models.gnn_layers import GINELayer, GINLayer
from gnnfree.utils.utils import *

class HomogeneousEdgeGraphModel(nn.Module):
    def __init__(self, num_layers, inp_dim, out_dim, edge_dim, drop_ratio=0, JK = 'last'):
        super().__init__()
        self.num_layers = num_layers
        self.drop_ratio = 0
        self.JK = JK

        self.conv = torch.nn.ModuleList()

        self.batch_norm = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.conv.append(GINELayer(inp_dim, out_dim, edge_dim))
            else:
                self.conv.append(GINELayer(out_dim, out_dim, edge_dim))
            self.batch_norm.append(torch.nn.BatchNorm1d(out_dim))

        self.timer = SmartTimer(False)

    def forward(self, g):
        h = g.ndata['feat']
        e = g.edata['feat']
        h_list = [h]
        for layer in range(self.num_layers):
            h = self.conv[layer](g, h_list[layer], e)
            h = self.batch_norm[layer](h)
            if layer != self.num_layers -1:
                h = F.relu(h)
            h_list.append(h)
        if self.JK == 'last':
            repr = h_list[-1]
        elif self.JK == 'sum':
            repr = 0
            for layer in range(self.num_layers+1):
                repr += h_list[layer]
        return repr


class HomogeneousGraphModel(nn.Module):
    def __init__(self, num_layers, inp_dim, out_dim, drop_ratio=0, JK = 'last'):
        super().__init__()
        self.num_layers = num_layers
        self.drop_ratio = 0
        self.JK = JK

        self.conv = torch.nn.ModuleList()

        self.batch_norm = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.conv.append(GINLayer(inp_dim, out_dim))
            else:
                self.conv.append(GINLayer(out_dim, out_dim))
            self.batch_norm.append(torch.nn.BatchNorm1d(out_dim))

        self.timer = SmartTimer(False)

    def forward(self, g):
        h = g.ndata['feat']
        h_list = [h]

        for layer in range(self.num_layers):
            h = self.conv[layer](g, h_list[layer])
            h = self.batch_norm[layer](h)
            if layer != self.num_layers -1:
                h = F.relu(h)
            h_list.append(h)
        
        if self.JK == 'last':
            repr = h_list[-1]
        elif self.JK == 'sum':
            repr = 0
            for layer in range(self.num_layers+1):
                repr += h_list[layer]
        return repr


class RGCN(nn.Module):
    def __init__(self, num_layers, num_rels, inp_dim, emb_dim, num_bases=4, dropout=0, JK='last'):
        super(RGCN, self).__init__()
        self.num_hidden_layers = num_layers
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.dropout = dropout
        self.JK = JK

        self.conv = torch.nn.ModuleList()

        self.batch_norm = torch.nn.ModuleList()

        for layer in range(self.num_hidden_layers):
            if layer == 0:
                self.conv.append(RelGraphConv(self.inp_dim, self.emb_dim, self.num_rels, num_bases=self.num_bases, activation=F.relu, dropout=self.dropout))
            else:
                self.conv.append(RelGraphConv(self.emb_dim, self.emb_dim, self.num_rels, num_bases=self.num_bases, activation=F.relu, dropout=self.dropout))
            self.batch_norm.append(torch.nn.BatchNorm1d(self.emb_dim))

        self.timer = SmartTimer(False)

    def forward(self, g):
        h = g.ndata['feat']
        h_list = [h]

        for layer in range(self.num_hidden_layers):
            h = self.conv[layer](g, h_list[layer], g.edata['type'])
            h = self.batch_norm[layer](h)
            h_list.append(h)
        
        if self.JK == 'last':
            repr = h_list[-1]
        elif self.JK == 'sum':
            repr = 0
            for layer in range(self.num_layers+1):
                repr += h_list[layer]
        return repr