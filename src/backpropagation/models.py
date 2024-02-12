import time
import torch
from torch import nn
from layers import BinaryLinear, BinaryLinearOutput


class AbstractModel(nn.Module):
    def __init__(self, name):
        super(AbstractModel, self).__init__()
        self.name = name
        self.layers = None

    def binarize(self):
        for layer in self.layers:
            layer.binarize()

    def clip(self):
        for layer in self.layers:
            layer.clip()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def train_and_save(self, device, train_loader, criterion, learning_rate=0.01, num_epochs=100):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        print("Start training", self.name)

        start = time.time()
        for epoch in range(num_epochs):
            loss_epoch = 0.
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()
                self.clip() # мапит веса в диапазон от -1 до 1
                optimizer.zero_grad()

                loss_epoch += loss.item()
                # print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}({n})], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch:.4f}')
        end = time.time() - start
        print("End training", self.name, "in", end/60, "minutes or", end/60/60, "hours")

        self.binarize() # бинаризует итоговые веса
        torch.save(self, "./data/models/backpropagation/" + self.name + ".pth")


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
