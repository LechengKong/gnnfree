import numpy as np
import scipy.sparse as ssp
import torch
import dgl

from graph_tool.all import Graph, random_shortest_path

def construct_graph_from_edges(ori_head, ori_tail, n_entities, inverse_edge=False, edge_type=None, num_rels=None):
    num_rels = 1
    if inverse_edge:
        head = np.concatenate([ori_head, ori_tail])
        tail = np.concatenate([ori_tail, ori_head])
    else:
        head = ori_head
        tail = ori_tail
    g = dgl.graph((head, tail), num_nodes=n_entities)
    g.edata['src_node'] = torch.tensor(head, dtype=torch.long)
    g.edata['dst_node'] = torch.tensor(tail, dtype=torch.long)
    if edge_type is not None:
        if num_rels is None:
            num_rels = np.max(edge_type)+1
        g.edata['type'] = torch.tensor(np.concatenate((edge_type, edge_type+num_rels)))
    return g


def dgl_graph_to_gt_graph(dgl_graph, directed=True):
    row, col = dgl_graph.edges()
    edges = torch.cat([row.view(-1,1), col.view(-1,1)],dim=-1)
    gt_g = Graph()
    gt_g.add_vertex(int(dgl_graph.num_nodes()))
    gt_g.add_edge_list(edges.numpy())
    gt_g.set_directed(directed)
    return gt_g

def remove_gt_graph_edge(gt_graph, s, t):
    edges = gt_graph.edge(s,t,all_edges=True)
    for e in edges:
        gt_graph.remove_edge(e)
    if gt_graph.is_directed():
        edges = gt_graph.edge(t,s,all_edges=True)
        for e in edges:
            gt_graph.remove_edge(e)

def add_gt_graph_edge(gt_graph, s, t):
    gt_graph.add_edge(s,t)
    if gt_graph.is_directed():
        gt_graph.add_edge(t,s)

