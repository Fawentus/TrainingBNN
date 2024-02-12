# Мы изначально ждём данные \in {-1, 1}, в статье также
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class OneHotColor:
    def __call__(self, sample):
        sample = sample.flatten()
        onehot_encoded = torch.empty(0)
        for p in sample:
            onehot = torch.full([256], -1)
            onehot[p.item()] = 1
            onehot_encoded = torch.cat((onehot, onehot_encoded))
        return onehot_encoded.flatten()


def create_MNIST(batch_size=100):
    train_dataset = MNIST("./data",
                          train=True,
                          download=True,
                          transform=transforms.Compose([transforms.PILToTensor(), OneHotColor()]))
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, batch_size=batch_size)

    test_dataset = MNIST("./data",
                         train=False,
                         download=True,
                         transform=transforms.Compose([transforms.PILToTensor(), OneHotColor()]))
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, drop_last=True, batch_size=batch_size)
    return train_loader, test_loader, 28*28*256, 10
