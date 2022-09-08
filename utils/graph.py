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

def generate_random_cycle_plus_graph(cycle_length, aug_nodes=3, sample_size=10, gen_p=0.2):
    head = np.arange(cycle_length)
    tail = np.arange(1,cycle_length+1)
    chain_edges = np.stack([head,tail]).T
    cycle_edges = np.concatenate([chain_edges,np.array([[cycle_length,0]])])
    num_nodes = cycle_length+1+aug_nodes
    aug_nodes_ind = np.arange(cycle_length+1, num_nodes)

    cycle_graphs = []
    chain_graphs = []

    for i in range(sample_size):
        aug_edge = []
        for j in aug_nodes_ind:
            p = np.random.binomial(1, p=np.zeros(num_nodes)+gen_p).nonzero()[0]
            for k in p:
                aug_edge.append([j, k])
        extra = np.random.permutation(num_nodes)[:2].reshape(1,2)
        edges = np.array(aug_edge)
        if len(edges)!=0:
            cycle_graphs.append(np.concatenate([cycle_edges, edges]))
            chain_graphs.append(np.concatenate([chain_edges, edges, extra]))

    data = []
    for g in cycle_graphs:
        cg = construct_graph_from_edges(g[:,0],g[:,1],n_entities=num_nodes,inverse_edge=True)
        data.append([cg,1])
    for g in chain_graphs:
        cg = construct_graph_from_edges(g[:,0],g[:,1],n_entities=num_nodes,inverse_edge=True)
        data.append([cg,0])
    return data

def generate_random_grid_graph(cycle_length, aug_nodes=3, sample_size=10, gen_p=0.2):
    a_edges = [[0,1],[0,2],[1,3],[2,3],[2,4],[3,5],[4,5]]
    b_edges = [[0,1],[0,2],[1,2],[2,3],[3,4],[3,5],[4,5]]
    chain_edges = np.array(a_edges)
    cycle_edges = np.array(b_edges)
    num_nodes = 6+aug_nodes
    aug_nodes_ind = np.arange(6, num_nodes)

    cycle_graphs = []
    chain_graphs = []

    for i in range(sample_size):
        aug_edge = []
        for j in aug_nodes_ind:
            p = np.random.binomial(1, p=np.zeros(num_nodes)+gen_p).nonzero()[0]
            for k in p:
                aug_edge.append([j, k])
        edges = np.array(aug_edge)
        if len(edges)!=0:
            cycle_graphs.append(np.concatenate([cycle_edges, edges]))
            chain_graphs.append(np.concatenate([chain_edges, edges]))

    data = []
    for g in cycle_graphs:
        cg = construct_graph_from_edges(g[:,0],g[:,1],n_entities=num_nodes,inverse_edge=True)
        data.append([cg,1])
    for g in chain_graphs:
        cg = construct_graph_from_edges(g[:,0],g[:,1],n_entities=num_nodes,inverse_edge=True)
        data.append([cg,0])
    return data

def generate_randomvv_grid_graph(cycle_length, aug_nodes=3, sample_size=10, gen_p=0.2):
    a_edges = [[0,1],[0,2],[1,3],[2,3],[2,4],[3,5],[4,5]]
    b_edges = [[0,1],[0,2],[1,2],[2,3],[3,4],[3,5],[4,5]]
    chain_edges = np.array(a_edges)
    cycle_edges = np.array(b_edges)
    num_nodes = 6+aug_nodes
    aug_nodes_ind = np.arange(6, num_nodes)

    cycle_graphs = []
    chain_graphs = []

    for i in range(sample_size):
        aug_edge = []
        for j in aug_nodes_ind:
            p = np.random.binomial(1, p=np.zeros(num_nodes)+gen_p).nonzero()[0]
            for k in p:
                aug_edge.append([j, k])
        edges = np.array(aug_edge)
        if len(edges)!=0:
            cycle_graphs.append(np.concatenate([cycle_edges, edges]))
        aug_edge = []
        for j in aug_nodes_ind:
            p = np.random.binomial(1, p=np.zeros(num_nodes)+gen_p).nonzero()[0]
            for k in p:
                aug_edge.append([j, k])
        edges = np.array(aug_edge)
        if len(edges)!=0:
            chain_graphs.append(np.concatenate([chain_edges, edges]))

    data = []
    for g in cycle_graphs:
        cg = construct_graph_from_edges(g[:,0],g[:,1],n_entities=num_nodes,inverse_edge=True)
        data.append([cg,1])
    for g in chain_graphs:
        cg = construct_graph_from_edges(g[:,0],g[:,1],n_entities=num_nodes,inverse_edge=True)
        data.append([cg,0])
    return data
