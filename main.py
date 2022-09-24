from logging import critical
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from qqdm import qqdm
import time
import os, random, numpy as np

start_time = time.time()

# fix random seed
print('seeded')
seed = 0
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "mps"
train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=128)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 10),
        )
    def forward(self, x):
        return self.net(x)

Net = Model().to(device)
optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)
critirion = nn.CrossEntropyLoss()

for epoch in range(10):
    progress_bar = qqdm(train_loader)
    for (x, y) in progress_bar:
        x = x.to(device)
        y = y.to(device)
        y_pred = Net(x)
        loss = critirion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_infos({
            'epoch': epoch,
            'loss': f'{loss:.4f}',
        })
        
n_correct = 0
n_sample = 0
for i, (x, y) in enumerate(train_loader):
    x = x.to(device)
    y = y.to(device)
    out = Net(x)
    _, pred = torch.max(out, 1)
    n_correct += (pred == y).sum().item()
    n_sample += y.shape[0]
acc = n_correct / n_sample

print(f'\nAccuracy: {acc:.4f}', f'total time when using {device}: {time.time() - start_time:.4f} seconds')