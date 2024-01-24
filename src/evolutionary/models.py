import time
import torch
from torch import nn
from src.evolutionary.layers import BinaryLinear, BinaryLinearOutput


class AbstractModel(nn.Module):
    def __init__(self, name):
        super(AbstractModel, self).__init__()
        self.name = name
        self.layers = None

    def binarize(self):  # бинаризует веса
        for layer in self.layers:
            layer.binarize()

    def evolutionary_step(self):  # генерирует новые веса
        for layer in self.layers:
            layer.evolutionary_step()

    def reset_weight(self):  # возвращает старые веса
        for layer in self.layers:
            layer.reset_weight()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def train_and_save(self, train_loader, criterion, learning_rate=0.01, num_epochs=100, num_steps_evolutionary=100):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        # n_total_steps = len(train_loader)
        n = 50  # это чтобы сократить время обучения

        print("Start training", self.name)
        start = time.time()
        for epoch in range(num_epochs):
            loss_epoch = 0.
            for i, (images, labels) in enumerate(train_loader):
                if i > n:  # это чтобы сократить время обучения
                    break

                # Forward pass
                loss = criterion(self(images), labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    # Эволюционный алгоритм
                    for j in range(num_steps_evolutionary):
                        self.binarize()
                        loss1 = criterion(self(images), labels)
                        self.evolutionary_step()
                        loss2 = criterion(self(images), labels)
                        if loss1 < loss2:
                            self.reset_weight()
                optimizer.zero_grad()

                loss_epoch = loss.item()
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}({n})], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch:.4f}')
            # print("end epoch", epoch, "-----------------------------------------")
        end = time.time() - start
        print("End training", self.name, "in", end / 60, "minutes or", end / 60 / 60, "hours")

        torch.save(self, "../../data/models/evolutionary/" + self.name + ".pth")


class Model1(AbstractModel):
    def __init__(self, input_dim, output_dim):
        super(Model1, self).__init__("model1")
        self.layers = nn.Sequential(
            BinaryLinear(input_dim, 16),
            BinaryLinear(16, 32),
            BinaryLinear(32, 64),
            BinaryLinear(64, 32),
            BinaryLinearOutput(32, output_dim),
        )


class Model2(AbstractModel):
    def __init__(self, input_dim, output_dim):
        super(Model2, self).__init__("model2")
        self.layers = nn.Sequential(
            BinaryLinear(input_dim, 16),
            BinaryLinearOutput(16, output_dim),
        )


# class AbstractModel(nn.Module):
#     def __init__(self, name):
#         super(AbstractModel, self).__init__()
#         self.name = name
#         self.layers = None
#
#     def binarize(self): # бинаризует веса
#         raise NotImplementedError
#
#     def evolutionary_step(self): # генерирует новые веса
#         raise NotImplementedError
#
#     def reset_weight(self): # возвращает старые веса
#         raise NotImplementedError
#
#     def train_and_save(self, train_loader, criterion, learning_rate=0.01, num_epochs=100, num_steps_evolutionary=100):
#         optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
#         # n_total_steps = len(train_loader)
#         n = 50 # это чтобы сократить время обучения
#
#         print("Start training", self.name)
#         start = time.time()
#         for epoch in range(num_epochs):
#             # print("start epoch", epoch, "-----------------------------------------")
#             loss_epoch = 0.
#             for i, (images, labels) in enumerate(train_loader):
#                 if i > n: # это чтобы сократить время обучения
#                     break
#
#                 # Forward pass
#                 loss = criterion(self(images), labels)
#
#                 # Backward and optimize
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 with torch.no_grad():
#                     # Эволюционный алгоритм
#                     for j in range(num_steps_evolutionary):
#                         # print("start step evolutionary", j, "-----------------------------------------")
#                         self.binarize()
#                         loss1 = criterion(self(images), labels)
#                         self.evolutionary_step()
#                         loss2 = criterion(self(images), labels)
#                         # print("loss1 < loss2", loss1 < loss2)
#                         if loss1 < loss2:
#                             self.reset_weight()
#                         # print("end step evolutionary", j, "-----------------------------------------")
#                 optimizer.zero_grad()
#
#                 loss_epoch = loss.item()
#                 # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}({n})], Loss: {loss.item():.4f}')
#             print(f'Epoch [{epoch + 1}/{num_epochs}]], Loss: {loss_epoch:.4f}')
#             # print("end epoch", epoch, "-----------------------------------------")
#         end = time.time() - start
#         print("End training", self.name, "in", end/60, "minutes or", end/60/60, "hours")
#
#         torch.save(self, "../../data/models/evolutionary/" + self.name + ".pth")
#
#
# class Model1(AbstractModel):
#     def __init__(self, input_dim, output_dim):
#         super(Model1, self).__init__("model1")
#         self.layer1 = BinaryLinear(input_dim, 16)
#         self.layer2 = BinaryLinear(16, 32)
#         self.layer3 = BinaryLinear(32, 64)
#         self.layer4 = BinaryLinear(64, 32)
#         self.layer5 = BinaryLinearOutput(32, output_dim)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         return self.layer5(out)
#
#     def binarize(self):
#         self.layer1.binarize()
#         self.layer2.binarize()
#         self.layer3.binarize()
#         self.layer4.binarize()
#         self.layer5.binarize()
#
#     def evolutionary_step(self):
#         self.layer1.evolutionary_step()
#         self.layer2.evolutionary_step()
#         self.layer3.evolutionary_step()
#         self.layer4.evolutionary_step()
#         self.layer5.evolutionary_step()
#
#     def reset_weight(self):  # возвращает старые веса
#         self.layer1.reset_weight()
#         self.layer2.reset_weight()
#         self.layer3.reset_weight()
#         self.layer4.reset_weight()
#         self.layer5.reset_weight()
#
#
# class Model2(AbstractModel):
#     def __init__(self, input_dim, output_dim):
#         super(Model2, self).__init__("model2")
#         self.layer1 = BinaryLinear(input_dim, 16)
#         self.layer2 = BinaryLinearOutput(16, output_dim)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         return self.layer2(out)
#
#     def binarize(self):
#         # print(self.name, "start binarize", "-----------------------------------")
#         self.layer1.binarize()
#         self.layer2.binarize()
#         # print(self.name, "end binarize", "-----------------------------------")
#
#     def evolutionary_step(self):
#         # print(self.name, "start evolutionary_step", "-----------------------------------")
#         self.layer1.evolutionary_step()
#         self.layer2.evolutionary_step()
#         # print(self.name, "end evolutionary_step", "-----------------------------------")
#
#     def reset_weight(self):  # возвращает старые веса
#         self.layer1.reset_weight()
#         self.layer2.reset_weight()
