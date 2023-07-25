import torch, torchvision 
from torch import nn
from torch import optim 
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

from sklearn.metrics import confusion_matrix
import pandas as pd 
import numpy as np 

import requests
from PIL import Image 
from io import BytesIO

def create_lenet():
    model = nn.Sequential(
        
        nn.Conv2d(1,6,5,padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        
        nn.Conv2d(6,16,5,padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        
        nn.Flatten(),
        nn.Linear(16*5*5,120),
        nn.ReLU(),
        nn.Linear(120,84), 
        nn.ReLU(),
        nn.Linear(84,10)
        
    )
    return model