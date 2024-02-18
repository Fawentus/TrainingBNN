import sys
import time
import torch
from torch import nn
from layers import BinaryLinear, BinaryLinearOutput
from heapq import nlargest


class AbstractModel(nn.Module):
    def __init__(self, name):
        super(AbstractModel, self).__init__()
        self.name = name
        self.layers = None

    def binarize(self):  # бинаризует веса
        for layer in self.layers:
            layer.binarize()

    def set_weights(self, i):
        for layer in self.layers:
            layer.set_weights(i)

    def genetic_select(self, ids):
        for layer in self.layers:
            layer.genetic_select(ids)

    def genetic_select_random(self, n, d):
        for layer in self.layers:
            layer.genetic_select_random(n, d)

    def genetic_crossover(self, m, d):
        for layer in self.layers:
            layer.genetic_crossover(m, d)

    def genetic_step(self):
        for layer in self.layers:
            layer.genetic_step()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def generate_population(self, size):
        for layer in self.layers:
            layer.generate_population(size)

    def train_and_save(self, device, train_loader, criterion, learning_rate=0.01, num_epochs=100, num_steps_genetic=100, stagnation=10, n=4, m=4, e=2):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        n_total_steps = len(train_loader)
        self.generate_population(n+m+e)

        print("Start training", self.name)
        start = time.time()
        for epoch in range(num_epochs):
            loss_epoch = 0.
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                loss = criterion(self(images), labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    # Генетический алгоритм
                    j = 0
                    is_end = 0
                    while j < num_steps_genetic or (num_steps_genetic == -1 and is_end < stagnation):
                        j += 1
                        self.binarize()
                        loss1 = criterion(self(images), labels)

                        f = []
                        sum_f = 0
                        for i in range(n+m+e):
                            self.set_weights(i) # устанавливает в качестве весов i-ого индивида
                            loss = criterion(self(images), labels)
                            f.append(loss)
                            sum_f += loss
                        d = []
                        for _ in range(n+m+e):
                            d.append(f[i] / sum_f)
                        e_id_max_val = list(map(lambda p: p[0], nlargest(e, enumerate(d), key=lambda pair: pair[1])))
                        self.genetic_select(e_id_max_val) # добавляет в следующую популяцию индивидов с переданными индексами
                        self.genetic_select_random(n, d) # добавляет в следующую популяцию n индивидов согласно переданной вероятности
                        self.genetic_crossover(m, d) # добавляет в следующую популяцию m индивидов с помощью кроссовера согласно переданной вероятности
                        self.genetic_step() # делает следующую популяцию текущей

                        f = []
                        for i in range(n+m+e): # не можем повторно использовать на следующем шаге, так как там изменятся b
                            self.set_weights(i) # устанавливает в качестве весов i-ого индивида
                            loss = criterion(self(images), labels)
                            f.append(loss)
                        i_max = max(enumerate(f), key=lambda pair: pair[1])[0]
                        self.set_weights(i_max)

                        loss2 = f[i_max]
                        if loss1 < loss2:
                            is_end += 1
                        else:
                            is_end = 0
                optimizer.zero_grad()

                loss_epoch += loss.item()
                if i % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}', file=sys.stderr)
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}({n})], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch:.4f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch:.4f}', file=sys.stderr)
            # print("end epoch", epoch, "-----------------------------------------")
        end = time.time() - start
        print("End training", self.name, "in", end / 60, "minutes or", end / 60 / 60, "hours")
        print("End training", self.name, "in", end / 60, "minutes or", end / 60 / 60, "hours", file=sys.stderr)

        torch.save(self, "./data/models/genetic/" + self.name + ".pth")


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
