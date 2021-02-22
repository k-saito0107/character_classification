import numpy as np
from PIL import Image
import cv2
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torchvision.datasets import ImageFolder
import pandas as pd
import torch.optim as optim

from utils.data_augumentation import Resize, Noise, Line_Noise
from utils.train import train


def main():
    mean = (0.5)
    std = (0.5)
    width = 448
    height = 224
    salt = [0.1, 0.4]
    papper = [0.01, 0.04]
    batch_size = 16

    train_transform = transforms.Compose([
        Resize(width, height),
        Noise(salt, papper),
        Line_Noise(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        Resize(width, height),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    train_dataset = ImageFolder('/kw_resources/character_classification/data/train_data', train_transform)
    val_dataset = ImageFolder('/kw_resources/character_classification/data/validation_data', val_transform)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=2)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False, num_workers=2)


    #model
    use_pretrained = False
    model = models.resnet34(pretrained=use_pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,stride=2, padding=3)
    model.fc = nn.Linear(in_features=512, out_features=44)

    num_epoch = 200
    up_model = train(model, num_epoch, train_loader, val_loader)







if __name__ == '__main__':
    main()
    print('finish')