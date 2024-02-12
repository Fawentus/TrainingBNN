import sys
import torch
from torch import nn
from dataset import create_MNIST
from models import Model1, Model2, Model3
from torchsummary import summary


def train(name, batch_size=100, num_epochs=10, num_steps_evolutionary=100):
    train_loader, _, input_size, output_size = create_MNIST(batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, file=sys.stderr)

    if name == "model1":
        criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
        model = Model1(input_size, output_size)
        model.to(device)
        model.train_and_save(device,
                             train_loader,
                             criterion,
                             num_epochs=num_epochs,
                             num_steps_evolutionary=num_steps_evolutionary)
    if name == "model2":
        criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
        model = Model2(input_size, output_size)
        model.to(device)
        model.train_and_save(device,
                             train_loader,
                             criterion,
                             num_epochs=num_epochs,
                             num_steps_evolutionary=num_steps_evolutionary)
    if name == "model3":
        criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
        model = Model3(input_size, output_size)
        model.to(device)
        model.train_and_save(device,
                             train_loader,
                             criterion,
                             num_epochs=num_epochs,
                             num_steps_evolutionary=num_steps_evolutionary)


def load(name, batch_size=100):
    # torch.set_printoptions(threshold=200)
    model = torch.load("./data/models/evolutionary/" + name + ".pth")
    model.eval()
    summary(model, input_size=(28 * 28 * 256,))
    for param in model.parameters():
        print("PARAM:", param.shape, param)

    _, test_loader, _, _ = create_MNIST(batch_size=batch_size)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            _, predicted = torch.max(model(images).data, 1) # max returns (value ,index)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


train("model3", num_epochs=200, num_steps_evolutionary=-1)
load("model3")
