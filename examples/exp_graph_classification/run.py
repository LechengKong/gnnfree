import torch
import os.path as osp
import numpy as np
import dgl
from gnnfree.managers.trainer import Trainer
from gnnfree.nn.loss import MultiClassLoss

from gnnfree.nn.models.GNN import HomogeneousGNN
from gnnfree.nn.models.task_predictor import GraphEncoder
from gnnfree.managers import Manager
from gnnfree.utils.datasets import DatasetWithCollate


from gnnfree.utils.evaluators import BinaryAccEvaluator
from gnnfree.utils.io import load_exp_dataset
from gnnfree.utils.utils import k_fold2_split, k_fold_ind, set_random_seed
from gnnfree.managers.learner import SingleModelLearner
from gnnfree.nn.models.basic_models import MLPLayers, Predictor


class GraphPredictionLearner(SingleModelLearner):
    def forward_func(self, batch):
        res = self.model(batch.g, batch)
        # print(res)
        return res

    def data_to_loss_arg(self, res, batch):
        return res, batch.labels


class GraphLabelDataset(DatasetWithCollate):
    def __init__(self, graphs, labels) -> None:
        super().__init__()

        self.graphs = graphs
        self.labels = labels

    def __getitem__(self, index):
        return self.graphs[index], np.array([self.labels[index]])

    def __len__(self):
        return len(self.graphs)

    def get_collate_fn(self):
        return collate_graph_label


class GraphLabelBatch:
    def __init__(self, samples) -> None:
        self.ls = []
        self.g_list = []
        self.label_list = []

        for g, label in samples:
            self.g_list.append(g)
            self.label_list.append(label)
        self.ls.append(dgl.batch(self.g_list))
        self.ls.append(torch.tensor(np.concatenate(self.label_list)))

    def to_name(self):
        self.g = self.ls[0]
        self.labels = self.ls[1]

    def to(self, device):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].to(device)
        self.device = device
        return self


def collate_graph_label(samples):
    return GraphLabelBatch(samples)


set_random_seed(1)

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")

graphs, label = load_exp_dataset(
    osp.join("./exp", "GRAPHSAT.txt")
)

pos_node_num = 5
rep = 1
emb_dim = 16

num_classes = 2

gnn = HomogeneousGNN(4, 2, emb_dim, batch_norm=True)

graph_encoder = GraphEncoder(emb_dim, gnn)

classifier = MLPLayers(1, [graph_encoder.get_out_dim(), num_classes])

graph_predictor = Predictor(graph_encoder, classifier).to(device)


loss = MultiClassLoss()
evlter = BinaryAccEvaluator("acc")


folds = k_fold_ind(label, fold=5)
splits = k_fold2_split(folds, len(label))
s = splits[0]


train = GraphLabelDataset([graphs[i] for i in s[0]], [label[i] for i in s[0]])
test = GraphLabelDataset([graphs[i] for i in s[1]], [label[i] for i in s[1]])
val = GraphLabelDataset([graphs[i] for i in s[2]], [label[i] for i in s[2]])

# data = GraphLabelDataset(graphs, label)


optimizer = torch.optim.Adam(graph_predictor.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

train_learner = GraphPredictionLearner(
    "train_gp_learner", train, graph_predictor, 32
)
val_learner = GraphPredictionLearner(
    "val_gp_learner", val, graph_predictor, 32
)
test_learner = GraphPredictionLearner(
    "test_gp_learner", test, graph_predictor, 32
)

manager = Manager()

trainer = Trainer(evlter, loss, 8)

manager.train(
    train_learner,
    val_learner,
    trainer,
    optimizer,
    "acc",
    device=device,
    num_epochs=100,
)

manager.load_model(train_learner, optimizer)

val_res = manager.eval(val_learner, trainer, device=device)
test_res = manager.eval(test_learner, trainer, device=device)
