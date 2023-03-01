import torch
import os.path as osp
import argparse
import networkx as nx
import numpy as np
import pickle as pkl
import dgl
from gnnfree.nn.loss import MultiClassLoss

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gnnfree.nn.models.pyg import PyGGIN
from gnnfree.nn.models.util_model import MLP
from dataset import GraphLabelDataset

from gnnfree.utils.evaluators import BinaryAccEvaluator
from gnnfree.utils.io import load_exp_dataset
from gnnfree.utils.utils import (
    k_fold2_split,
    k_fold_ind,
    set_random_seed,
    dict_res_summary,
)
from gnnfree.utils.graph import construct_graph_from_edges

from model import PlainLabelGNN

from lightning_models import GraphRandLabelCls

from utils import setup_exp


def main(params):
    setup_exp(params)

    # -------------------------- Data Preparation --------------------|
    graphs, label = load_exp_dataset("./exp/GRAPHSAT.txt")
    params.inp_dim = 2
    params.num_class = 2

    folds = k_fold_ind(label, fold=5)
    splits = k_fold2_split(folds, len(label))
    s = splits[0]

    train = GraphLabelDataset(
        [graphs[i] for i in s[0]], [label[i] for i in s[0]]
    )
    test = GraphLabelDataset(
        [graphs[i] for i in s[1]], [label[i] for i in s[1]]
    )
    val = GraphLabelDataset(
        [graphs[i] for i in s[2]], [label[i] for i in s[2]]
    )

    # ----------------------------- Data Preparation End -------------|
    params.rep = 1

    task_gnn = PyGGIN(
        params.num_layers,
        params.num_piece + params.inp_dim,
        params.emb_dim,
        batch_norm=True,
    )
    graph_encoder = PlainLabelGNN(params.emb_dim, task_gnn)

    classifier = MLPLayers(
        3,
        [graph_encoder.get_out_dim(), 512, 512, params.num_class],
        batch_norm=True,
    )

    loss = MultiClassLoss()
    evlter = BinaryAccEvaluator("acc")

    eval_metric = "acc"

    graph_pred = GraphRandLabelCls(
        params.exp_dir,
        {"train": train, "test": test, "val": val},
        params,
        task_gnn,
        graph_encoder,
        classifier,
        params.num_piece,
        loss,
        evlter,
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=params.num_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            ModelCheckpoint(monitor=eval_metric, mode="max"),
        ],
        logger=CSVLogger(save_dir=params.exp_dir),
    )
    trainer.fit(graph_pred)
    # trainer.test()
    # params.rep = 3

    test_col = []
    for i in range(10):
        test_col.append(trainer.test()[0])

    print(
        np.mean(dict_res_summary(test_col)[eval_metric]),
        np.std(dict_res_summary(test_col)[eval_metric]),
    )

    # graphs = []
    # label = []
    # r_label = []
    # res = []
    # for j in ind:
    #     g = nx_graphs[j]
    #     edges = np.array(list(g.edges))
    #     for i in range(25):
    #         for k in range(25):
    #             g = construct_graph_from_edges(
    #                 edges[:, 0], edges[:, 1], n_entities=25, inverse_edge=True
    #             )
    #             g.ndata["feat"] = torch.ones((25, 1))
    #             graphs.append(g.to(graph_pred.device))
    #             label.append(j)
    #             r_label.append(
    #                 torch.tensor([[i, k]], dtype=torch.long).to(
    #                     graph_pred.device
    #                 )
    #             )
    # graph_pred.eval()
    # for i in range(int(len(graphs) / params.batch_size) + 1):
    #     g = dgl.batch(
    #         graphs[i * params.batch_size : (i + 1) * params.batch_size]
    #     )
    #     lbs = torch.cat(
    #         r_label[i * params.batch_size : (i + 1) * params.batch_size], dim=0
    #     )
    #     res.append(graph_pred.model(g, lbs, None))
    # scores = torch.cat(res, dim=0).detach().cpu().numpy()
    # scores = scores.reshape(params.num_class, 25, 25, params.num_class)
    # for i in range(params.num_class):
    #     mat = scores[i]
    #     res = np.argmax(mat, axis=-1)
    #     val = np.bincount(res.flatten(), minlength=15)
    #     print(i, val, val[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gnn")

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="srg")

    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--mol_emb_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--JK", type=str, default="last")
    parser.add_argument("--hidden_dim", type=int, default=32)

    parser.add_argument("--num_piece", type=int, default=2)

    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)

    parser.add_argument("--psearch", type=bool, default=False)

    params = parser.parse_args()
    set_random_seed(1)
    main(params)
