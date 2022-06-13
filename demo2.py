import tensorflow as tf
import numpy as np

import cv2
import os

tf.config.list_physical_devices('GPU')

dataset = []
images = []
masks = []

path = r'E:\MiniProj\Image_segmentation\cityscapes_data\train\\'
image_dest = r'E:\MiniProj\Image_segmentation\car dataset\train\Images\\'
mask_dest = r'E:\MiniProj\Image_segmentation\car dataset\train\Masks\\'

i=0
for image_path in os.listdir(path):
    image_path = path+image_path
    i=i+1
    image = cv2.imread(image_path)
    # image = cv2.resize(image,(740,740))
    cv2.imshow('12', image)
    cv2.waitKey(1)
    img_np = np.array(image)
    image = img_np[:, :256, :]
    mask = img_np[:, 256:, :]
tf.keras.

    filename = image_dest+str(i)+'.jpg'
    cv2.imwrite(filename, image)


    filename = mask_dest+str(i)+'.jpg'
    cv2.imwrite(filename, mask)

