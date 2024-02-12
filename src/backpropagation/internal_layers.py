import math
import torch.nn as nn
import torch.nn.functional as Function
from torch.autograd import Function as Grad


class BinaryGrad(Grad):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.detach().clone()
        grad_input[abs(grad_output) <= 1] = 1
        grad_input[abs(grad_output) > 1] = 0
        return grad_input


def clip(x, y, z):
    # max(y, min(z, x))
    assert z > y
    res = x.detach().clone()
    res[x > z] = z
    res[x < y] = y
    return res


class BinaryInternalLinear(nn.Linear):
    def forward(self, input):
        binary_weight = BinaryGrad.apply(self.weight)
        return Function.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Xavier Glorot and Yoshua Bengio "Understanding the difficulty of training deep feedforward neural networks", 2010
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(6. / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        self.weight.lr_scale = 1. / stdv

    def binarize(self):
        self.weight = nn.Parameter(BinaryGrad.apply(self.weight))

    def clip(self):
        self.weight = nn.Parameter(clip(self.weight, -1, 1))


class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, input):
        return BinaryGrad.apply(input)
