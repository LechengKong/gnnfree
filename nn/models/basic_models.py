import torch.nn as nn

class MLPLayers(nn.Module):
    def __init__(self, layers, h_units, dropout=0, batch_norm=True, activation=nn.ReLU):
        super().__init__()
        self.mlp = nn.Sequential()
        self.activation = activation
        for i in range(layers):
            if i>0:
                if batch_norm:
                    self.mlp.append(nn.BatchNorm1d(h_units[i+1]))
                self.mlp.append(self.activation())
                if dropout>0:
                    self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(nn.Linear(h_units[i], h_units[i+1]))
            
    def forward(self, x):
        return self.mlp(x)

