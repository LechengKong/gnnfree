import numpy as np
import scipy.sparse as ssp
import torch
import dgl

def construct_graph_from_edges(ori_head, ori_tail, n_entities, inverse_edge=False, edge_type=None):
    num_rels = 1
    if inverse_edge:
        head = np.concatenate([ori_head, ori_tail])
        tail = np.concatenate([ori_tail, ori_head])
    g = dgl.graph((head, tail), num_nodes=n_entities)
    g.edata['src_node'] = torch.tensor(head, dtype=torch.long)
    g.edata['dst_node'] = torch.tensor(tail, dtype=torch.long)
    if edge_type is not None:
        num_rels = np.max(edge_type)+1
        g.edata['type'] = torch.tensor(np.concatenate((edge_type, edge_type+num_rels)))
    return g
