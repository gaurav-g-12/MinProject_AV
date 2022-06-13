import os
import tensorflow as tf
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_addons as tfa
import datetime


tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)

seed = 42
path = 'E:\MiniProj\Image_segmentation\car dataset'


def image_and_mask(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(image_path, "Images", "Masks")

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

    return {'image':image, 'mask':mask}


train_dataset = tf.data.Dataset.list_files(path + '\\' + 'train' + '\\' + 'Images' +'\\' + '*.jpg', seed=seed)
train_dataset = train_dataset.map(image_and_mask)
val_dataset = tf.data.Dataset.list_files(path + '\\' + 'val' + '\\' + 'Images'+ '\\' + '*.jpg', seed=seed)
val_dataset = val_dataset.map(image_and_mask)

dataset = {"train": train_dataset, "val": val_dataset}


def resize_norm(datapoint):
    image = tf.image.resize(datapoint['image'], (128, 128))
    mask = tf.image.resize(datapoint['mask'], (128, 128))

    image = tf.cast(image, tf.float32)/255.0


    # image = tf.image.flip_left_right(image)
    # mask = tf.image.flip_left_right(mask)

    return image, mask

dataset['train'] = dataset['train'].map(resize_norm, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .repeat()\
    .shuffle(buffer_size=1000, seed=seed)\
    .batch(batch_size=32)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


dataset['val'] = dataset['val'].map(resize_norm).repeat()\
    .batch(batch_size=32)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# print(dataset['train'])

def display_func(list):
    plt.figure(figsize=(10,10))
    for i in range(len(list)):
        plt.subplot(1,3,i+1)
        plt.imshow(list[i])
        plt.axis('off')
    plt.show()

for image, mask in dataset['train'].take(1):
    sample_image, sample_mask = image, mask

display_func([sample_image[0], sample_mask[0]])

def encoder(inputs):
  base_model = tf.keras.applications.MobileNetV2(input_shape=(128,128,3), include_top=False)
  base_model.trainable = False

  layers = ['block_1_expand_relu',
            'block_3_expand_relu',
            'block_6_expand_relu',
            'block_13_expand_relu',
            'block_16_expand_relu']

  skip_layers = [base_model.get_layer(name).output for name in layers]

  down_stacker = tf.keras.Model(inputs = base_model.input, outputs = skip_layers)(inputs)

  return down_stacker

def decoder():
  up_stacker = [pix2pix.upsample(512, 3),
               pix2pix.upsample(256, 3),
               pix2pix.upsample(128, 3),
               pix2pix.upsample(64, 3)]

  return up_stacker

def unet_model():

  inputs = tf.keras.layers.Input(shape=[128,128,3])

  skips = encoder(inputs)

  x = skips[-1]

  skips = reversed(skips[:-1])

  up_sampler = decoder()

  for up, skip in zip(up_sampler, skips):
    x = up(x)
    concatenate = tf.keras.layers.Concatenate()
    x = concatenate([x, skip])

  last_layer = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same')

  x = last_layer(x)

  return tf.keras.Model(inputs = inputs, outputs = x)


final_model = unet_model()

final_model.summary()

# final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss = tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])


def create_mask(pred_mask):

    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def show_pred(pred):
    for images, masks in dataset['train'].take(1):
        pred_mask = final_model.predict(images)
        final_mask = create_mask(pred_mask)
        display_func([images[0], masks[0],final_mask[0]])


show_pred(dataset)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


EPOCHS = 20

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [
    # to show samples after each epoch
    DisplayCallback(),
    # to collect some useful metrics and visualize them in tensorboard
    tensorboard_callback,
    # if no accuracy improvements we can stop the training directly
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    # to save checkpoints
    tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
]



# # here I'm using a new optimizer: https://arxiv.org/abs/1908.03265
optimizer=tfa.optimizers.RectifiedAdam(lr=0.1)
loss = tf.keras.losses.SparseCategoricalCrossentropy()


final_model.compile(optimizer=optimizer, loss = loss,
                  metrics=['accuracy'])


model_history = final_model.fit(dataset['train'], epochs=EPOCHS,
                    steps_per_epoch=2000,
                    validation_steps=20,
                    validation_data=dataset['val'],
                    callbacks=callbacks)




