import time
import torch
from torch import nn
from layers import BinaryLinear, BinaryLinearOutput


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

    def train_and_save(self, train_loader, criterion, learning_rate=0.01, num_epochs=100, num_steps_evolutionary=100, stagnation=10):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        print("Start training", self.name)

        start = time.time()
        for epoch in range(num_epochs):
            loss_epoch = 0.
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                loss = criterion(self(images), labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    # Эволюционный алгоритм
                    j = 0
                    is_end = 0
                    while j < num_steps_evolutionary or (num_steps_evolutionary == -1 and is_end < stagnation):
                        j += 1
                        self.binarize()
                        loss1 = criterion(self(images), labels)
                        self.evolutionary_step()
                        loss2 = criterion(self(images), labels)
                        if loss1 < loss2:
                            self.reset_weight()
                            is_end += 1
                        else:
                            is_end = 0
                optimizer.zero_grad()

                loss_epoch += loss.item()
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}({n})], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch:.4f}')
            # print("end epoch", epoch, "-----------------------------------------")
        end = time.time() - start
        print("End training", self.name, "in", end / 60, "minutes or", end / 60 / 60, "hours")

        torch.save(self, "./data/models/evolutionary/" + self.name + ".pth")


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


class Model3(AbstractModel):
    def __init__(self, input_dim, output_dim):
        super(Model3, self).__init__("model3")
        self.layers = nn.Sequential(
            BinaryLinear(input_dim, 512),
            BinaryLinear(512, 512),
            BinaryLinear(512, 128),
            BinaryLinearOutput(128, output_dim),
        )
