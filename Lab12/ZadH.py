import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from keras.models import Seqeuntial
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import History

from keras.applications import VGG16, InceptionV3
base_model = VGG16(weights='imagenet',include_top=False)
base_model.summary()

inc = InceptionV3()
inc.summary()

h,w = 32, 32
model = VGG16(weights='imagenet',include_top=False,input_shape=(h,w,3))

model.summary()

h,w = 32, 32
model = VGG16(weights='imagenet',include_top=False,input_shape=(h,w,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(4, activation='sigmoid'))

model_transfer = Sequential()
model_transfer.add(model)
model_transfer.add(top_model)

model_transfer.layers[0].trainable = False

model_transfer.summary()

from keras.datasets import cifar10
# from scipy.misc import toimage

import numpy as np

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

nb_classes = 4
n_samples = 7000

X_train, y_train = X_train[np.where(y_train<nb_classes)[0]][:n_samples], y_train[np.where(y_train<nb_classes)[0]][:n_samples]
X_test, y_test = X_test[np.where(y_test<nb_classes)[0]], y_test[np.where(y_test<nb_classes)[0]]

print(X_train.shape)
print(X_test.shape)
print(np.unique(y_train,return_counts=True))
print(X_train[0].shape)


# normalize inputs from 0-255 to 0.0-1.0

X_train = X_train/255
X_test = X_test/255

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             rotation_range=90,
                             zoom_range=[0.5, 1.0],
                             brightness_range=[0.1, 1.5])
it = datagen.flow(X_train, batch_size=32)

# one hot encode outputs
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_classes = y_test.shape[1]

model_transfer.compile(optimizer="adam",
                       loss="categorical_crossentropy", metrics=["accuracy"])

model_transfer.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, batch_size=32, validation_split=0.2)

print(model_transfer.evaluate(X_test, y_test))