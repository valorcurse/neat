import torch
import torchvision
import torchvision.transforms as transforms

if torch.cuda.is_available():
    print("Using CUDA, number of devices = ",torch.cuda.device_count())
else:
    print("Can't find CUDA")

import matplotlib.pyplot as plt
import numpy as np


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.kernelSize = 3
        self.convDepth = 3
        padding = 0

        self.conv1 = nn.Conv2d(3, 6, self.kernelSize, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, self.kernelSize, padding=padding)
        # self.conv3 = nn.Conv2d(128, 128, self.kernelSize, padding=padding)
        # self.conv4 = nn.Conv2d(6, 16, self.kernelSize, padding=padding)
        
        # self.fc1 = nn.Linear(16 * 3 * 3, 120)
        self.fc1 = nn.Linear(16 * 76 * 158, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = x.cuda()
        # print("------------------------")
        # print("------------------------")
        # print(x.size())
        # print(self.conv1(x).size())
        # print(F.relu(self.conv1(x)).size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())

        # print("1------------------------")
        
        # print(self.conv2(x).size())
        # print(F.relu(self.conv2(x)).size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())

        # print("2------------------------")

        # print(x.size())
        x = x.view(-1, 16 * 76 * 158)
        # print(x.size())

        # print("3------------------------")

        # print(self.fc1(x).size())
        x = F.relu(self.fc1(x))
        # print(x.size())

        # print("4------------------------")

        # print(self.fc2(x).size())
        x = F.relu(self.fc2(x))
        # print(x.size())
        x = self.fc3(x)
        return x