import math
import torch
import torch.nn as nn
from torch.autograd import Function as Grad


class BinaryGrad(Grad):
    @staticmethod
    def forward(cxt, input):
        output = torch.empty_like(input)
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.detach().clone()
        grad_input[abs(grad_output) <= 1] = 1
        grad_input[abs(grad_output) > 1] = 0
        return grad_input


class BinaryInternalLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.old_weight = None

    def reset_parameters(self):
        # Xavier Glorot and Yoshua Bengio "Understanding the difficulty of training deep feedforward neural networks", 2010
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(6. / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        self.weight.lr_scale = 1. / stdv

    def binarize(self):
        weight = torch.empty_like(self.weight)
        weight[self.weight >= 0] = 1.
        weight[self.weight < 0] = -1.
        self.weight = nn.Parameter(weight)

    def reset_weight(self):
        assert self.old_weight is not None
        self.weight = nn.Parameter(self.old_weight)

    def evolutionary_step(self):
        self.eval()
        self.old_weight = self.weight.detach().clone()

        curr_weight = self.weight.detach().clone()
        b = []
        numel_weight = torch.numel(self.weight)
        while 1. not in b:
            b = torch.bernoulli(torch.full(self.weight.shape, 1/numel_weight))
        assert 1. in b

        b_1 = b.detach().clone()
        b_m1 = b.detach().clone()
        b_1[curr_weight == -1.] = 0.
        b_m1[curr_weight == 1.] = 0.

        curr_weight[b_m1 == 1.] = 1.
        curr_weight[b_1 == 1.] = -1.

        self.weight = nn.Parameter(curr_weight)
        self.train()


class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, input):
        return BinaryGrad.apply(input)
