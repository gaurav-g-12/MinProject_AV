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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

np.random.seed(44)
root_path = r'E:\MiniProj\Image_segmentation\cityscapes_data'

data_dir = root_path
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)

sample_image_path = os.path.join(train_dir, train_fns[70])
sample_image = Image.open(sample_image_path).convert("RGB")
# cv2.imshow('window', np.array(sample_image))
# cv2.waitKey(0)

def split_image(image):
    image = np.array(image)
    cityscape, label = image[:, :256, :], image[:, 256:, :]
    return cityscape, label

sample_image = np.array(sample_image)
cityscape, label = split_image(sample_image)

# cityscape, label = Image.fromarray(cityscape), Image.fromarray(label)
# cv2.imshow('window1', cityscape)
# cv2.imshow('window2', label)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

num_items = 1000
color_array = np.random.choice(range(0, 256,3), 3*num_items).reshape(-1,3)

num_classes = 10
kmeans_model = KMeans(n_clusters = num_classes)
kmeans_model.fit(color_array)

cityscape, label = split_image(sample_image)
label_class = kmeans_model.predict(label.reshape(-1,3)).reshape(256,256)

# cv2.imshow('wim1', np.array(cityscape))
# cv2.imshow('wim2', np.array(label))
#
# plt.imshow(label_class)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()


class CityscapeDataset(Dataset):

  def __init__(self, image_dir, label_model):
    self.image_dir = image_dir
    self.image_name_list = os.listdir(image_dir)
    self.label_model = label_model

  def __len__(self):
    return len(self.image_name_list)

  def __getitem__(self, index):
    image_name = self.image_name_list[index]
    image_path = os.path.join(self.image_dir, image_name)
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    cityscape, label = self.split_image(image)
    label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
    label_class = torch.Tensor(label_class).long()
    cityscape = self.transform(cityscape)
    return cityscape, label_class

  def split_image(self, image):
    image = np.array(image)
    cityscape, label = image[:, :256, :], image[:, 256:, :]
    return cityscape, label

  def transform(self, image):
    transform_ops = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform_ops(image)

dataset = CityscapeDataset(train_dir, kmeans_model)

cityscape, label_class = dataset[0]

# cv2.imshow('win', cityscape)
# plt.imshow(label_class)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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


model = UNet(num_classes=1000)



batch_size = 5
epochs = 10
lr = 0.01

dataset = CityscapeDataset(train_dir, kmeans_model)
data_loader = DataLoader(dataset, batch_size = batch_size)

model = UNet(num_classes = num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)


step_losses = []
epoch_losses = []

# for epoch in (range(epochs)):
#   epoch_loss = 0
#   for X,Y in tqdm(data_loader, total=len(data_loader), leave = False):
#     X, Y = X.to(device), Y.to(device)
#     optimizer.zero_grad()
#     Y_pred = model(X)
#     loss = criterion(Y_pred, Y)
#     loss.backward()
#     optimizer.step()
#     epoch_loss += loss.item()
#     step_losses.append(loss.item())
#   epoch_losses.append(epoch_loss/len(data_loader))



# fig, axes = plt.subplots(1,2, figsize=(10,5))
# axes[0].plot(step_losses)
# axes[1].plot(epoch_losses)


num_classes=10
image_path = r'E:\MiniProj\1.png'

model_path = r'E:\MiniProj\Image_segmentation\cityscapes_dataUNet.pth'
model_ = UNet(num_classes = num_classes).to(device)
model_.load_state_dict(torch.load(model_path))

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])


image = Image.open(image_path)
image1 = loader(image).float()
image1 = Variable(image1, requires_grad=True)
image1 = image1.unsqueeze(0)
image1 = image1.cuda()



Y_pred = model_(image1)
Y_pred = torch.argmax(Y_pred, dim=1)
print(Y_pred[0].shape)

label = Y_pred[0].cpu()
image = np.array(image)
# label = np.array(Y_pred[0])
cv2.imshow('win', image)
plt.imshow(label)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
































