import networkx as nx

from torch.utils.data import Dataset

class DatasetWithCollate(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def get_collate_fn(self):
        raise NotImplementedError

class SingleGraphDataset(DatasetWithCollate):
    def __init__(self, graph):
        super().__init__()
        self.num_nodes = graph.num_nodes()
        
        self.graph = graph
        self.adj_mat = self.graph.adjacency_matrix(transpose=False, scipy_fmt='csr')
        self.nx_graph = nx.from_scipy_sparse_matrix(self.adj_mat)
