from gnnfree.nn.models.GNN import MultiLayerMessagePassing

from torch_geometric.nn.models import MLP
from dgl.nn.pytorch.conv import GINConv, RelGraphConv


class DGLGIN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.build_layers()

    def build_input_layer(self):
        return GINConv(
            MLP(
                [self.inp_dim, 2 * self.inp_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            learn_eps=True,
        )

    def build_hidden_layer(self):
        return GINConv(
            MLP(
                [self.out_dim, 2 * self.out_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            learn_eps=True,
        )

    def build_message_from_input(self, g):
        return {"g": g, "h": g.ndata["feat"]}

    def build_message_from_output(self, g, h):
        return {"g": g, "h": h}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["g"], message["h"])


class DGLRGCN(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers,
        num_rels,
        inp_dim,
        out_dim,
        num_bases=4,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.build_layers()

    def build_input_layer(self):
        return RelGraphConv(
            self.inp_dim, self.out_dim, self.num_rels, num_bases=self.num_bases
        )

    def build_hidden_layer(self):
        return RelGraphConv(
            self.out_dim, self.out_dim, self.num_rels, num_bases=self.num_bases
        )

    def build_message_from_input(self, g):
        return {"g": g, "h": g.ndata["feat"], "e": g.edata["type"]}

    def build_message_from_output(self, g, h):
        return {"g": g, "h": h, "e": g.edata["type"]}

    def layer_forward(self, layer, message):
        return self.conv[layer](message["g"], message["h"], message["e"])
