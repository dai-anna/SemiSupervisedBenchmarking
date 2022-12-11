"""
It is the neural network projection head g(·) that maps representations to 
the space where contrastive loss is applied. There is the option to use a linear
projection head or a nonlinear projection head (2 layer MLP).
"""


import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bnorm=False, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.bnorm = bnorm

        self.linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias and not self.bnorm,
        )
        if self.bnorm:
            self.bnorm = nn.BatchNorm1d(self.out_features)

    def forward(self, x):
        x = self.linear(x)
        self.bnorm(x) if self.bnorm else x
        return x


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        head_type="nonlinear",
        **kwargs
    ):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == "linear":
            self.layers = LinearLayer(
                in_features=self.in_features,
                out_features=self.out_features,
                bias=False,
                bnorm=True,
            )
        elif self.head_type == "nonlinear":
            self.layers = nn.Sequential(
                LinearLayer(
                    in_features=self.in_features,
                    out_features=self.hidden_features,
                    bias=True,
                    bnorm=True,
                ),
                nn.ReLU(),
                LinearLayer(
                    in_features=self.hidden_features,
                    out_features=self.out_features,
                    bias=False,
                    bnorm=True,
                ),
            )

    def forward(self, x):
        x = self.layers(x)
        return x
