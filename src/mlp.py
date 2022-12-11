"""
It is the neural network projection head g(·) that maps representations to 
the space where contrastive loss is applied. There is the option to use a linear
projection head or a nonlinear projection head (2 layer MLP).
"""

import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=False, use_bn=False):
        super(LinearLayer, self).__init__()

        self.use_bn = use_bn
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=use_bias and not use_bn,
        )
        self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x) if self.use_bn else x
        return x


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        head_type="nonlinear",
        use_bn=True,
    ):
        super(ProjectionHead, self).__init__()

        if head_type == "linear":
            self.layers = LinearLayer(
                in_features=in_features,
                out_features=out_features,
                use_bias=False,
                use_bn=True,
            )
        elif head_type == "nonlinear":
            self.layers = nn.Sequential(
                LinearLayer(
                    in_features=in_features,
                    out_features=hidden_features,
                    use_bias=True,
                    use_bn=use_bn,
                ),
                nn.ReLU(),
                LinearLayer(
                    in_features=hidden_features,
                    out_features=out_features,
                    use_bias=False,
                    use_bn=use_bn,
                ),
            )

    def forward(self, x):
        x = self.layers(x)
        return x
