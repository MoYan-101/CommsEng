"""
models/model_ann.py

Implementation of an ANN for multi-output regression with optional Dropout.
"""
import torch
import torch.nn as nn

class ANNRegression(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[32,64,32],
                 dropout=0.0,
                 activation="ReLU",
                 random_seed=None):
        """
        :param input_dim: input feature dimension
        :param output_dim: output dimension
        :param hidden_dims: list of hidden-layer sizes
        :param dropout: dropout probability, for example 0.2
        :param activation: activation name, for example "ReLU" / "Tanh" / "Sigmoid"
        :param random_seed: if set, use this seed to initialize network weights
        """
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)

        act_fn = None
        if activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid
        elif activation.lower() == "leakyrelu":
            act_fn = lambda: nn.LeakyReLU(negative_slope=0.005)
        else:
            act_fn = nn.ReLU

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
