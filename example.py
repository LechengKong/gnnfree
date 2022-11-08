import torch
import os.path as osp
from gnnfree.managers.trainer import MaxTrainer
from gnnfree.nn.loss import MultiClassLoss

from gnnfree.nn.models.GNN import HomogeneousGNN
from gnnfree.nn.models.task_predictor import GraphClassifier
from gnnfree.managers.manager import Manager
from dgl.nn.pytorch.conv import GraphConv

from cano_dataset import CanoCommonDataset
from gnnfree.utils.evaluators import BinaryAccEvaluator
from graph_utils import load_exp_dataset
from learners import GraphPredictionLearner

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")


graphs, label = load_exp_dataset(osp.join("./data/exp", "GRAPHSAT.txt"))

data = CanoCommonDataset(graphs, label, 10)

gnn = HomogeneousGNN(3, 3, 8, layer_t=GraphConv)

clsifer = GraphClassifier(2, 8, gnn, add_self_loop=True).to(device)

loss = MultiClassLoss()
evlter = BinaryAccEvaluator("acc")


def out2evaldata(res, data):
    return [res, data.labels]


optimizer = torch.optim.Adam(clsifer.parameters(), lr=0.001)

lrner = GraphPredictionLearner(
    "train_gp_learner", data, clsifer, loss, optimizer, 8
)

manager = Manager()

trainer = MaxTrainer(evlter, out2evaldata, 8)

manager.train(lrner, lrner, trainer, optimizer, "acc", device=device)
