import torch, torchvision 
from torch import nn
from torch import optim 
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import requests
from PIL import Image 
from io import BytesIO

# from https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

def create_upgraded_lenet():
    model = nn.Sequential(
    nn.Conv2d(1, 32, 5, stride=1, padding=2),
    nn.ReLU(),
    nn.Conv2d(32, 32, 5, stride=1, padding=2, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Dropout(0.25),
    nn.Conv2d(32, 64, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Dropout(0.25),
    nn.Flatten(),
    nn.Linear(64*7*7, 256, bias=False),  # Adjusted this line
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 128, bias=False),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 84, bias=False),
    nn.BatchNorm1d(84),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(84, 10),
    nn.Softmax(dim=1)
    )
    return model

def get_optimizer_for_upgraded_lenet(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005)
    return optimizer
