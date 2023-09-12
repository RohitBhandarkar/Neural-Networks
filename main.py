import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class NN(nn.Module):
    def __init__(self, input_pixels, digits):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_pixels, 50)
        self.fc2 = nn.Linear(50, digits)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# model = NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x))
# print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_pixels = 784
digits = 10
learning_rate = 0.001
batch_size = 64
epochs = 1

train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_pixels=input_pixels, digits=digits).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch_id, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        data = data.reshape(data.shape[0], -1)  # reshape

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()  # gradient = 0, to avoid using gradient of previous forward propagations
        loss.backward()

        optimizer.step()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy of training data")
    else:
        print("checking accuracy of testing data")
    correct = samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            correct += (predictions == y).sum()
            samples += predictions.size(0)

        acc = (float(correct) / float(samples)) * 100
        print(f"Accuracy: {acc:.2f}")

    model.train()
    return acc


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
