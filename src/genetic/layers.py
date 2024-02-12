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

    def set_weights(self, i):
        self.layer[0].set_weights(i)

    def genetic_select(self, ids):
        self.layer[0].genetic_select(ids)

    def genetic_select_random(self, n, d):
        self.layer[0].genetic_select_random(n, d)

    def genetic_crossover(self, m, d):
        self.layer[0].genetic_crossover(m, d)

    def genetic_step(self):
        self.layer[0].genetic_step()

    def generate_population(self, size):
        self.layer[0].generate_population(size)


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

    def set_weights(self, i):
        self.layer[0].set_weights(i)

    def genetic_select(self, ids):
        self.layer[0].genetic_select(ids)

    def genetic_select_random(self, n, d):
        self.layer[0].genetic_select_random(n, d)

    def genetic_crossover(self, m, d):
        self.layer[0].genetic_crossover(m, d)

    def genetic_step(self):
        self.layer[0].genetic_step()

    def generate_population(self, size):
        self.layer[0].generate_population(size)
