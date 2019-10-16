import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
import PIL

EPOCH = 10
BATCH_SIZE = 4
LR = 0.001
TRAIN_DATA_PATH = "D:/user/Desktop/cs-ioc5008-hw1/dataset/dataset/train"
CROPSIZE = 200

transform_img = transforms.Compose([
    transforms.RandomResizedCrop(CROPSIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform_img)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# default network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32*48*48, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13)
    
    def forward(self, x):
        out = self.mp(self.relu(self.conv1(x)))
        out = self.mp(self.relu(self.conv2(out)))
        out = out.view(x.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

net = Net()
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

def trainData():
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_data_loader):   # gives batch data, normalize x when iterate train_loader
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output = net(b_x)            # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            
            if step % 50 == 0:
                loss.data = loss.data.cpu()
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
    torch.save(net, 'D:/user/Desktop/lenet.pkl')
    
if __name__ == '__main__':
    trainData()

   

  