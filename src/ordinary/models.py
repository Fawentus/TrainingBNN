import sys
import time
import torch
from torch import nn


class AbstractModel(nn.Module):
    def __init__(self, name):
        super(AbstractModel, self).__init__()
        self.name = name

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def train_and_save(self, device, train_loader, criterion, learning_rate=0.01, num_epochs=100):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        n_total_steps = len(train_loader)
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
                optimizer.zero_grad()

                loss_epoch += loss.item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch:.4f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch:.4f}', file=sys.stderr)
        end = time.time() - start
        print("End training", self.name, "in", end / 60, "minutes or", end / 60 / 60, "hours")
        print("End training", self.name, "in", end / 60, "minutes or", end / 60 / 60, "hours", file=sys.stderr)

        torch.save(self, "./data/models/ordinary/" + self.name + ".pth")


class Model1(AbstractModel):
    def __init__(self, input_dim, output_dim):
        super(Model1, self).__init__("model1")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),

            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),

            nn.Linear(32, output_dim),
        )


class Model2(AbstractModel):
    def __init__(self, input_dim, output_dim):
        super(Model2, self).__init__("model2")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),

            nn.Linear(16, output_dim),
        )


class Model3(AbstractModel):
    def __init__(self, input_dim, output_dim):
        super(Model3, self).__init__("model3")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),

            nn.Linear(128, output_dim),
        )
