import tensorflow as tf
from tensorflow import keras


from numpy.random import seed
seed(123)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from keras.datasets import cifar10

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

# one hot encode outputs
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_classes = y_test.shape[1]

from keras.models import Sequential
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

model = Sequential([
    Conv2D(filters=6, input_shape=(32, 32, 3),
           kernel_size=(3,3), padding="same"),
    AveragePooling2D(pool_size=(2,2)),
    Conv2D(filters=16, kernel_size=(2,2)),
    AveragePooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(120, activation="relu"),
    Dense(84, activation="relu"),
    Dense(nb_classes, activation="sigmoid")
])

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

history_1 = model.fit(X_train, y_train,
                      validation_split=0.2, epochs=15)

print(model.evaluate(X_test, y_test))

plt.plot(history_1.history['accuracy'], label = "tarina Adam")
plt.plot(history_1.history['val_accuracy'], label = "test Adam")


plt.legend()
plt.show()

