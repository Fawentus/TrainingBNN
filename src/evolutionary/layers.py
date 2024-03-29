import torch.nn as nn
from internal_layers import BinaryInternalLinear, Activation


class BinaryLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BinaryLinear, self).__init__()
        self.layer = nn.Sequential(
            BinaryInternalLinear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            Activation()
        )

    def forward(self, x):
        return self.layer(x)

    def binarize(self):
        self.layer[0].binarize()

    def evolutionary_step(self):
        self.layer[0].evolutionary_step()

    def reset_weight(self):  # возвращает старые веса
        self.layer[0].reset_weight()


class BinaryLinearOutput(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BinaryLinearOutput, self).__init__()
        self.layer = nn.Sequential(
            BinaryInternalLinear(input_dim, output_dim),
            # nn.Softmax()
        )

    def forward(self, x):
        return self.layer(x)

    def binarize(self):
        self.layer[0].binarize()

    def evolutionary_step(self):
        self.layer[0].evolutionary_step()

    def reset_weight(self):  # возвращает старые веса
        self.layer[0].reset_weight()
