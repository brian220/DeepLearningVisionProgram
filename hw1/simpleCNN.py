import os
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
from PIL import Image

EPOCH = 10
BATCH_SIZE = 10
LR = 0.001
TRAIN_DATA_PATH = "D:/user/Desktop/cs-ioc5008-hw1/dataset/dataset/train"
TEST_DATA_PATH = "D:/user/Desktop/cs-ioc5008-hw1/dataset/dataset/test/test"
CROPSIZE = 128

transform_img = transforms.Compose([
    transforms.RandomResizedCrop(CROPSIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5] )
    ])


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        #print(idx)
        img_name = os.listdir(self.root_dir)[idx]
        img_dir = os.path.join(self.root_dir, img_name)
        image = Image.open(img_dir).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        sample = (image, img_name)
        
        return sample


train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform_img)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_data = TestDataset(root_dir=TEST_DATA_PATH, transform=transform_img)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

classes = ('bedroom', 'coast', 'forest', 'highway','insidecity', 
           'kitchen', 'livingroom', 'mountain', 'office','opencountry',
           'street', 'suburb', 'tallbuilding')

# default network
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(    
                in_channels=3, # black white
                out_channels=32, # filter numbers
                kernel_size=5, # filter size
                stride=1,
                padding=2 # padding = (kernal_size-1)/2= (5 - 1)/2
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        self.out = torch.nn.Linear(16 * 32 * 32, 13) # hidden layer
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)      
        x = x.view(x.size(0), -1)
        output = self.out(x)
        
        return output


cnn = CNN()
cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted


def trainData():
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_data_loader):   # gives batch data, normalize x when iterate train_loader
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output = cnn(b_x)            # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 50 == 0:
                loss.data = loss.data.cpu()
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
                
    torch.save(cnn, 'cnn.pkl')

    
predictList = []
nameList = []
def testData():
    cnnTest = torch.load('cnn.pkl')
    for data in test_data_loader:
        images, name = data
        test_output = cnnTest(Variable(images))
        pred_y = torch.max(test_output, 1).cuda()
        predictList.extend(pred_y[1].data.numpy().tolist())
        nameList.extend(name)
 

if __name__ == '__main__':
    trainData()
    """testData()
    with open(r'D:/user/Desktop/output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for label, name in  zip(predictList, nameList):           
            writer.writerow([name[:-4], classes[label]])"""
  
