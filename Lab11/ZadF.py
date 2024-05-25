import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

input_shape = (28, 28)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

sgd = keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

Adam1 = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
Adam2 = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

optimizers = {'Sgd' : sgd, 'Adam LR 0.001':Adam1, 'Adam LR 0.0001': Adam2}

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.losses import sparse_categorical_crossentropy
for optimizer in optimizers:
    model = Sequential()
    model.add(Flatten(input_shape=(input_shape)))
    model.add(Dense(100,activation="sigmoid"))
    model.add(Dense(50,activation="sigmoid"))
    model.add(Dense(10,activation="sigmoid"))
    model.add(Dense(10,activation="sigmoid"))
    model.summary()

    model.compile(loss=sparse_categorical_crossentropy,optimizer=optimizers[optimizer],
                  metrics=["accuracy"])
    history = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32,epochs=100)

    plt.plot(history.history['accuracy'], label = "train")
    plt.plot(history.history['val_accuracy'], label = "test")
    plt.title(optimizer + 'accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label = "train")
    plt.plot(history.history['val_loss'], label = "test")
    plt.title(optimizer + 'loss')
    plt.legend()
    plt.show()

