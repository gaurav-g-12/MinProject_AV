import os
import tensorflow as tf
import numpy as np
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

tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)

seed = 42
path = 'E:\Datasets\cityscapes_data'

def image_and_mask(image_path):
    image = tf.io.read_file(image_path)
    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.uint8)

    cityscape, label = image[:, :256, :], image[:, 256:, :]
    return cityscape, label


train_dataset = tf.data.Dataset.list_files(path + '\\' + 'train' +'\\' + '*.jpg', seed=seed)
train_dataset = train_dataset.map(image_and_mask)
val_dataset = tf.data.Dataset.list_files(path + '\\' + 'val' + '\\' + '*.jpg', seed=seed)
val_dataset = val_dataset.map(image_and_mask)

dataset = {"train": train_dataset, "val": val_dataset}


num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1,3)
print(color_array.shape)

num_classes = 10
label_model = KMeans(n_clusters = num_classes)
label_model.fit(color_array)

sample_image = train_dataset[0]['label']

cityscape, label = split_image(sample_image)
label_class = label_model.predict(label.reshape(-1,3)).reshape(256,256)
cv2.imshow('wim1', np.array(cityscape))
cv2.imshow('wim2', np.array(label))
cv2.imshow('wim3', np.array(label_class))
cv2.waitKey(0)
cv2.destroyAllWindows()

# datapoint is a array containing label and cityscape
def resize_norm(datapoint):
    image = tf.image.resize(datapoint['cityscape'], (128, 128))
    mask = tf.image.resize(datapoint['label'], (128, 128))

    image = tf.cast(image, tf.float32)/255.0

    return image, mask

dataset['train'] = dataset['train'].map(resize_norm, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .repeat()\
    .shuffle(buffer_size=1000, seed=seed)\
    .batch(batch_size=5)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


dataset['val'] = dataset['val'].map(resize_norm).repeat()\
    .batch(batch_size=5)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print(dataset['train'])

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


dropout_rate = 0.5
input_size = (128, 128, 3)
N_CLASSES = 151
initializer = 'he_normal'

# -- Encoder -- #
# Block encoder 1
inputs = Input(shape=input_size)
conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
conv_enc_1 = Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

# Block encoder 2
max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

# Block  encoder 3
max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)

# Block  encoder 4
max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)
# -- Encoder -- #

# ----------- #
maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
# ----------- #

# -- Dencoder -- #
# Block decoder 1
up_dec_1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

# Block decoder 2
up_dec_2 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

# Block decoder 3
up_dec_3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

# Block decoder 4
up_dec_4 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
# -- Dencoder -- #

output = Conv2D(N_CLASSES, 1, activation = 'softmax')(conv_dec_4)



def create_mask(pred_mask):

    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def show_pred(pred):
    for images, masks in dataset['train'].take(1):
        pred_mask = final_model.predict(images)
        final_mask = create_mask(pred_mask)
        display_func([images[0], masks[0],final_mask[0]])

final_model = tf.keras.Model(inputs = inputs, outputs = output)
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
    show_pred(dataset),
    # to collect some useful metrics and visualize them in tensorboard
    tensorboard_callback,
    # if no accuracy improvements we can stop the training directly
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    # to save checkpoints
    tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

optimizer = tfa.optimizers.RectifiedAdam(lr=1e-3)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

model = tf.keras.Model(inputs = inputs, outputs = output)

model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy'])

model_history = model.fit(dataset['train'], epochs=EPOCHS,
                    steps_per_epoch=200,
                    validation_steps=200,
                    validation_data=dataset['val'],
                    callbacks=callbacks)




















































