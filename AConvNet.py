import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.getcwd()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as isin

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps")
device

mnist_train = datasets.MNIST(root="../Data/", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = datasets.MNIST(root="../Data/", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

batch_size = 100
learning_rate = 0.0002
num_epoch = 10

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)




class AConvNet(nn.Module):
    def __init__(self):
        super(AConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.avgpool = nn.AvgPool2d(3, stride=2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.bn1(out)
        out = torch.square(out)
        out = self.avgpool(out)
        out = torch.log(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
model = AConvNet().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




loss_arr =[]
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y = label.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(x)
        
        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()
        
        if j % 1000 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())
            
            
            
            
correct = 0
total = 0

# evaluate modelmodel.eval()

with torch.no_grad():
    for image,label in test_loader:
        x = image.to(device)
        y= label.to(device)

        output = model.forward(x)
        
        # torch.max함수는 (최댓값,index)를 반환 
        _,output_index = torch.max(output,1)
        
        # 전체 개수 += 라벨의 개수
        total += label.size(0)
        
        # 도출한 모델의 index와 라벨이 일치하면 correct에 개수 추가
        correct += (output_index == y).sum().float()
    
    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))