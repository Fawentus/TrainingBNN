import math
import torch
import torch.nn as nn
from torch.autograd import Function as Grad
import random


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
        self.population = None
        self.next_population = []

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

    def generate_population(self, size):
        self.binarize()
        self.population = []
        for i in range(size):
            individual = self.weight.detach().clone()
            self.population.append(individual)

    def set_weights(self, i):  # устанавливает в качестве весов i-ого индивида
        assert self.population is not None
        assert len(self.population) > i

        self.weight = nn.Parameter(self.population[i])

    def genetic_select(self, ids):  # добавляет в следующую популяцию индивидов с переданными индексами
        assert self.population is not None

        for i in ids:
            individual = self.population[i].detach().clone()
            self.next_population.append(individual)

    def genetic_select_random(self, n,
                              d):  # добавляет в следующую популяцию n индивидов согласно переданной вероятности
        ids = random.choices(list(range(0, len(self.population))), weights=d, k=n)
        self.genetic_select(ids)

    def genetic_crossover(self, m,
                          d):  # добавляет в следующую популяцию m индивидов с помощью кроссовера согласно переданной вероятности
        for _ in range(m):
            parents = random.choices(list(range(0, len(self.population))), weights=d, k=2)
            point1 = random.randint(0, len(self.population[parents[0]]) - 1)
            point2 = random.randint(0, len(self.population[parents[0]]) - 1) + 1

            individual = self.population[parents[0]].detach().clone()
            for i in range(point1, point2):
                individual[i] = self.population[parents[1]][i]

            self.next_population.append(individual)

    def genetic_step(self):  # делает следующую популяцию текущей
        self.population = self.next_population
        self.next_population = []


class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, input):
        return BinaryGrad.apply(input)
