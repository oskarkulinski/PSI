import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import History, LearningRateScheduler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

input_shape = (X_train.shape[1], X_train.shape[2])

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

from keras.losses import sparse_categorical_crossentropy
### Adam no Scheduler

history_lr_1 = History()
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.summary()

adam1 = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=sparse_categorical_crossentropy,optimizer=adam1,
              metrics=["accuracy"])

model.fit(X_train, y_train, validation_data= (X_test, y_test),
          batch_size=32,epochs=100, callbacks=[history_lr_1])


#### Adam LR 0.001 scheduler

history_lr_2 = History()
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

adam2 = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=sparse_categorical_crossentropy,optimizer=adam2,
              metrics=["accuracy"])

lrate = LearningRateScheduler(step_decay)
model.fit(X_train, y_train, validation_data= (X_test, y_test),
          batch_size=32,epochs=100, callbacks=[lrate, history_lr_2])

#### Adam LR 0.0001 Scheduler

history_lr_3 = History()
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

adam3 = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss=sparse_categorical_crossentropy,optimizer=adam3,
              metrics=["accuracy"])

lrate = LearningRateScheduler(step_decay)
model.fit(X_train, y_train, validation_data= (X_test, y_test),
          batch_size=32,epochs=100, callbacks=[lrate, history_lr_3])

plt.plot(history_lr_1.history['accuracy'], label = "train Adam 0.001")
plt.plot(history_lr_1.history['val_accuracy'], label = "test Adam 0.001")
plt.plot(history_lr_2.history['accuracy'], label = "train Adam Scheduler 0.001")
plt.plot(history_lr_2.history['val_accuracy'], label = "test Adam Scheduler 0.001")
plt.plot(history_lr_3.history['accuracy'], label = "train Adam Scheduler 0.0001")
plt.plot(history_lr_3.history['val_accuracy'], label = "test Adam Scheduler 0.0001")
plt.legend()
plt.show()




