import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm

class UNet(nn.Module):

  def __init__(self, num_classes):
    super(UNet, self).__init__()
    self.num_classes = num_classes
    self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
    self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
    self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
    self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
    self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.middle = self.conv_block(in_channels=512, out_channels=1024)
    self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
    self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
    self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
    self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
    self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
    self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
    self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
    self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
    self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

  def conv_block(self, in_channels, out_channels):
    block = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(num_features=out_channels),
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(num_features=out_channels))
    return block

  def forward(self, X):
    contracting_11_out = self.contracting_11(X)  # [-1, 64, 256, 256]
    contracting_12_out = self.contracting_12(contracting_11_out)  # [-1, 64, 128, 128]
    contracting_21_out = self.contracting_21(contracting_12_out)  # [-1, 128, 128, 128]
    contracting_22_out = self.contracting_22(contracting_21_out)  # [-1, 128, 64, 64]
    contracting_31_out = self.contracting_31(contracting_22_out)  # [-1, 256, 64, 64]
    contracting_32_out = self.contracting_32(contracting_31_out)  # [-1, 256, 32, 32]
    contracting_41_out = self.contracting_41(contracting_32_out)  # [-1, 512, 32, 32]
    contracting_42_out = self.contracting_42(contracting_41_out)  # [-1, 512, 16, 16]
    middle_out = self.middle(contracting_42_out)  # [-1, 1024, 16, 16]
    expansive_11_out = self.expansive_11(middle_out)  # [-1, 512, 32, 32]
    expansive_12_out = self.expansive_12(
      torch.cat((expansive_11_out, contracting_41_out), dim=1))  # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
    expansive_21_out = self.expansive_21(expansive_12_out)  # [-1, 256, 64, 64]
    expansive_22_out = self.expansive_22(
      torch.cat((expansive_21_out, contracting_31_out), dim=1))  # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
    expansive_31_out = self.expansive_31(expansive_22_out)  # [-1, 128, 128, 128]
    expansive_32_out = self.expansive_32(
      torch.cat((expansive_31_out, contracting_21_out), dim=1))  # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
    expansive_41_out = self.expansive_41(expansive_32_out)  # [-1, 64, 256, 256]
    expansive_42_out = self.expansive_42(
      torch.cat((expansive_41_out, contracting_11_out), dim=1))  # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
    output_out = self.output(expansive_42_out)  # [-1, num_classes, 256, 256]
    return output_out

cap = cv2.VideoCapture(r'E:\MiniProj\Image_segmentation\test.mp4')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

num_classes=10

model_path = r'E:\MiniProj\Image_segmentation\cityscapes_dataUNet.pth'
model_ = UNet(num_classes = num_classes).to(device)
model_.load_state_dict(torch.load(model_path))

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

while(True):
  _,image = cap.read()
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image=cv2.resize(image, (256, 256))
  image =  Image.fromarray(image)
  image1 = loader(image).float()
  image1 = Variable(image1, requires_grad=True)
  image1 = image1.unsqueeze(0)
  image1 = image1.cuda()

  Y_pred = model_(image1)
  Y_pred = torch.argmax(Y_pred, dim=1)

  label = Y_pred[0].cpu()
  # label = np.expand_dims(label, axis=2)
  label = np.uint8(np.array(label))
  # label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
  label = cv2.applyColorMap(label, cv2.COLORMAP_HOT)

  print(label.shape)
  image = np.array(image)
  cv2.imshow('win', image)
  cv2.imshow('label ', label)
  # plt.imshow(label)
  # plt.show()
  if cv2.waitKey(1) & 0xff == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()












