"""
It is the neural network model base encoder f(·) that extracts representation vectors 
from augmented data examples. We choose to use the ResNet20 architecture.
"""

import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes > planes or stride > 1:
            # Option B
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity if self.downsample is None else self.downsample(identity)
        out = F.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, planes=16, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=32, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=64, blocks=layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        for i in range(blocks):
            layers.append(
                block(
                    inplanes=(self.inplanes if i == 0 else planes),
                    planes=planes,
                    stride=stride if i == 0 else 1,
                )
            )
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
