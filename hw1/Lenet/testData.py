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
from trainData import Net, BATCH_SIZE, transform_img

torch.manual_seed(1) 

TEST_DATA_PATH = "D:/user/Desktop/cs-ioc5008-hw1/dataset/dataset/test/test"


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        img_name = os.listdir(self.root_dir)[idx]
        img_dir = os.path.join(self.root_dir, img_name)
        image = Image.open(img_dir).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        sample = (image, img_name)
        return sample


test_data = TestDataset(root_dir=TEST_DATA_PATH, transform=transform_img)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False,num_workers=4)

classes = ('bedroom', 'coast', 'forest', 'highway','insidecity', 
           'kitchen', 'livingroom', 'mountain', 'office','opencountry',
           'street', 'suburb', 'tallbuilding')


predictList = []
nameList = []
def testData():
    testNet = torch.load('D:/user/Desktop/lenet.pkl')
    for data in test_data_loader:
        images, name = data
        test_output = testNet(Variable(images).cuda())
        pred_y = torch.max(test_output, 1)
        predictList.extend(pred_y[1].data.cpu().numpy().tolist())
        nameList.extend(name)


def writeCsv():
    with open(r'D:/user/Desktop/output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for label, name in  zip(predictList, nameList):           
            writer.writerow([name[:-4], classes[label]]) 


if __name__ == '__main__':
    testData()
    writeCsv()
    