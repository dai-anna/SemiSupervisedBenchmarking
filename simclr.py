import torch.nn as nn
from mlp import ProjectionHead
from resnet20 import ResNet, BasicBlock


class SimCLRModel(nn.Module):
    """
    Creates SimCLR model given encoder
    Args:
      encoder (nn.Module): Encoder
      projection_n_in (int): Number of input features of the projection head
      projection_n_hidden (int): Number of hidden features of the projection head
      projection_n_out (int): Number of output features of the projection head
      projection_use_bn (bool): Whether to use batch norm in the projection head
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_n_in: int = 2048,
        projection_n_hidden: int = 2048,
        projection_n_out: int = 2048,
        projection_use_bn: bool = True,
    ):
        super().__init__()

        self.encoder = ResNet(BasicBlock, [3, 3, 3])
        self.projection = ProjectionHead(
            in_features=projection_n_in,
            hidden_features=projection_n_hidden,
            out_features=projection_n_out,
            head_type="nonlinear",
            use_bn=projection_use_bn,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        return x
