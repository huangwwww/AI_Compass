#The first attempt: to use two convolutional layers

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, func):
        super(Net, self).__init__()
        self.func = func
        self.conv1 = nn.Conv2d(3, 10, 11)  # num of channel in, num of kernels, size of kernel
        self.pool = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(10, 10, 11)
        self.fc1 = nn.Linear(1440, 800)  ################## need to change
        self.fc2 = nn.Linear(800, 1)

    def forward(self, x):

        if self.func == 'sigmoid':
            #print(x.shape)
            x=x.permute(0,3,1,2)
            #print(x.shape)
            x = self.pool(F.sigmoid(self.conv1(x)))
            #print(x.shape)
            x = self.pool(F.sigmoid(self.conv2(x)))
            #print(x.shape)
            x = x.view(-1,1440)  ################## need to change
            x = F.sigmoid(self.fc1(x))
            x = self.fc2(x)
            x = x.squeeze(1)
            #print(x.shape)

        if self.func == 'tanh':
            x=x.permute(0,3,1,2)
            #print(x.shape)
            x = self.pool(F.tanh(self.conv1(x)))
            #print(x.shape)
            x = self.pool(F.tanh(self.conv2(x)))
            #print(x.shape)
            x = x.view(-1,1440)  ################## need to change
            x = F.tanh(self.fc1(x))
            x = self.fc2(x)
            x = x.squeeze(1)
            print(x.shape)

        if self.func == 'relu':
            #print(x.shape)
            x=x.permute(0,3,1,2)
            #print(x.shape)
            x = self.pool(F.relu(self.conv1(x)))
            #print(x.shape)
            x = self.pool(F.relu(self.conv2(x)))
            #print(x.shape)
            x = x.view(-1,1440)  ################## need to change
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = x.squeeze(1)
            #print(x.shape)
        return x
