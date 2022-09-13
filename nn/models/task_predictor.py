import torch
import torch.nn as nn
import dgl

from gnnfree.nn.models.basic_models import MLPLayers
from gnnfree.utils.utils import SmartTimer
from gnnfree.nn.pooling import NodePairExtractor, Pooler, feature_module_dict


class GraphClassifier(nn.Module):
    def __init__(self, num_classes, emb_dim, gnn, add_self_loop=False):
        super().__init__()

        self.gnn = gnn
        self.add_self_loop = add_self_loop

        self.timer = SmartTimer(False)

        self.mlp_graph = MLPLayers(1, h_units=[emb_dim, num_classes])

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp_graph.reset_parameters()

    def forward(self, g, input):

        if self.add_self_loop:
            bnodes = g.batch_num_nodes()
            bedges = g.batch_num_edges()
            g = dgl.add_self_loop(g)
            g.set_batch_num_nodes(bnodes)
            g.set_batch_num_edges(bedges)

        repr = self.gnn(g)

        g_repr = self.pool_from_graph(g, repr, input)

        return self.mlp_graph(g_repr)

    def pool_from_graph(self, g, repr, input):
        g.ndata['node_res'] = repr

        g_sum = dgl.sum_nodes(g, 'node_res')

        return g_sum

class LinkPredictor(nn.Module):
    def __init__(self, emb_dim, gnn, add_self_loop=False):
        super().__init__()

        self.emb_dim = emb_dim

        self.gnn = gnn
        
        self.timer = SmartTimer(False)

        self.node_pair_extract = NodePairExtractor(self.emb_dim)

        self.add_self_loop = add_self_loop

        self.link_dim = self.node_pair_extract.get_out_dim()

        self.use_only_embedding = False

        self.build_predictor()
        self.link_mlp = MLPLayers(3, [self.link_dim, self.emb_dim, self.emb_dim, 1])

    def build_predictor(self):
        pass

    def embedding_only_mode(self, state=True):
        self.use_only_embedding = state

    def forward(self, g, head, tail, input):
        if self.use_only_embedding:
            repr = g.ndata['repr']
        else:
            if self.add_self_loop:
                g = dgl.add_self_loop(g)
            repr = self.process_graph(g)
        res = self.predict_link(repr, head, tail, input)
        return res

    def process_graph(self, g):
        repr = self.gnn(g)
        return repr

    def predict_link(self, repr, head, tail, input):
        repr_list = []
        repr_list.append(self.node_pair_extract(repr, head, tail))
        g_rep = torch.cat(repr_list, dim=1)
        output = self.link_mlp(g_rep)

        return output


class NodeClassifier(nn.Module):
    def __init__(self, num_classes, emb_dim, gnn, add_self_loop=False):
        super().__init__()

        self.gnn = gnn
        self.add_self_loop = add_self_loop

        self.timer = SmartTimer(False)

        self.use_only_embedding = False

        self.mlp_node = MLPLayers(1, h_units=[emb_dim, num_classes])

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp_graph.reset_parameters()

    def embedding_only_mode(self, state=True):
        self.use_only_embedding = state

    def forward(self, g, node, input):
        if self.use_only_embedding:
            repr = g.ndata['repr']
        else:
            if self.add_self_loop:
                g = dgl.add_self_loop(g)
            repr = self.process_graph(g)

        node_repr = self.pool_node(g, repr, node, input)

        return self.mlp_node(node_repr)

    def process_graph(self, g):
        repr = self.gnn(g)
        return repr

    def pool_node(self, g, repr, node, input):
        return repr[node]