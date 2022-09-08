# GNNFREE
Extendable, easy to use pytorch Graph Neural Network package based on DGL.

## Use case

GNNFREE is a Graph Algorithm Design platform in which Data preparation, GNN design, downstream task prediction and training pipeline are decoupled so that users can easily plug-and-play their own ideas and methods on the platform.

GNNFREE provides well-defined classes for each component and users can extend the generic framework by subclassing the base classes.

Meanwhile, GNNFREE collects various graph utility functions including large-scale max-depth BFS algorithm, scattered reduce/split algorithms unseen from other packages to aid the design and implementation of graph algorithms.

## Example

```python

graphs, label = load_exp_dataset(osp.join('./data/exp', 'GRAPHSAT.txt'))

data = CanoCommonDataset(graphs, label, 10)

gnn = HomogeneousGNN(3, 3, 8, layer_t=GraphConv)

clsifer = GraphClassifier(2, 8, gnn, add_self_loop=True).to(device)

loss = MultiClassLoss()
evlter = BinaryAccEvaluator('acc')

def out2evaldata(res, data):
    return [res, data.labels]

optimizer = torch.optim.Adam(clsifer.parameters(), lr=0.001)

lrner = GraphPredictionLearner('train_gp_learner', data, clsifer, loss, optimizer, 8)

manager = Manager()

trainer = MaxTrainer(evlter, out2evaldata, 8)

manager.train(lrner, lrner, trainer, optimizer, 'acc', device=device)
```
