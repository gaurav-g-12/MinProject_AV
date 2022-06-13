import os
import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import cv2
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import UpSampling2D
import tensorflow_addons as tfa
import datetime
from sklearn.cluster import KMeans
from PIL import Image

tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)

seed = 44

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



def CityscapeDataset(image_path):
  print(image_path)
  image = Image.open(image_path).convert("RGB")
  image = np.array(image)
  cityscape, label = image[:, :256, :], image[:, 256:, :]
  label_class = label_model.predict(label.reshape(-1, 3)).reshape(256, 256)

  return cityscape, label_class

train_dataset = tf.data.Dataset.list_files(root_path + '\\' + 'train'  +'\\' + '*.jpg', seed=seed)
print(train_dataset)
train_dataset = train_dataset.map(CityscapeDataset)
val_dataset = tf.data.Dataset.list_files(path + '\\' + 'val' + '\\'  + '*.jpg', seed=seed)
val_dataset = val_dataset.map(CityscapeDataset)

dataset = {"train": train_dataset, "val": val_dataset}


def resize_norm(datapoint):
    image = tf.image.resize(datapoint['image'], (128, 128))
    mask = tf.image.resize(datapoint['mask'], (128, 128))
    return image, mask

dataset['train'] = dataset['train'].map(resize_norm, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .repeat()\
    .shuffle(buffer_size=1000, seed=seed)\
    .batch(batch_size=32)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


dataset['val'] = dataset['val'].map(resize_norm).repeat()\
    .batch(batch_size=32)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



for image, mask in dataset['train'].take(1):
    sample_image, sample_mask = image[0], mask[0]
    cv2.imshow('w', sample_image)
    plt.imshow(sample_mask)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# def encoder(inputs):
#   base_model = tf.keras.applications.MobileNetV2(input_shape=(128,128,3), include_top=False)
#   base_model.trainable = False
#
#   layers = ['block_1_expand_relu',
#             'block_3_expand_relu',
#             'block_6_expand_relu',
#             'block_13_expand_relu',
#             'block_16_expand_relu']
#
#   skip_layers = [base_model.get_layer(name).output for name in layers]
#
#   down_stacker = tf.keras.Model(inputs = base_model.input, outputs = skip_layers)(inputs)
#
#   return down_stacker
#
# def decoder():
#   up_stacker = [pix2pix.upsample(512, 3),
#                pix2pix.upsample(256, 3),
#                pix2pix.upsample(128, 3),
#                pix2pix.upsample(64, 3)]
#
#   return up_stacker
#
# def unet_model():
#
#   inputs = tf.keras.layers.Input(shape=[128,128,3])
#
#   skips = encoder(inputs)
#
#   x = skips[-1]
#
#   skips = reversed(skips[:-1])
#
#   up_sampler = decoder()
#
#   for up, skip in zip(up_sampler, skips):
#     x = up(x)
#     concatenate = tf.keras.layers.Concatenate()
#     x = concatenate([x, skip])
#
#   last_layer = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same')
#
#   x = last_layer(x)
#
#   return tf.keras.Model(inputs = inputs, outputs = x)
#
#
# final_model = unet_model()
#
# final_model.summary()
#
#
# optimizer=tfa.optimizers.RectifiedAdam(lr=0.1)
# loss = tf.keras.losses.SparseCategoricalCrossentropy()
#
#
# final_model.compile(optimizer=optimizer, loss = loss,
#                   metrics=['accuracy'])
#
#
# model_history = final_model.fit(dataset, epochs=1,
#                     steps_per_epoch=2000,
#                     validation_steps=20)















