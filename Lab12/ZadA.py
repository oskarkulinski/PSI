import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from numpy.random import seed
seed(123)


import numpy as np
import pandas as pd
import os
import keras.utils.np_utils
from sklearn.datasets import fetch_lfw_people

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names

# przygotowanie y
y = keras.utils.to_categorical(y)

n_classes = y.shape[1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

X_train = np.array([x.reshape((h, w, 1)) for x in X_train])
X_test = np.array([x.reshape((h, w, 1)) for x in X_test])
print(X_train.shape)


# skalowanie X

X_train = X_train/255
X_test = X_test/255

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten
from keras.callbacks import History

history_dense_1 = History()
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(n_classes,activation="softmax"))
model.summary()

early_stopping = EarlyStopping(patience=20,monitor="val_loss")
model.compile(loss="categorical_crossentropy",optimizer="adam",
              metrics=["categorical_accuracy"])
model.fit(X_train, y_train, validation_split=0.25,epochs=100,
          callbacks=[early_stopping, history_dense_1])

print(model.evaluate(X_test,y_test))

plt.plot(history_dense_1.history['categorical_accuracy'], label = "train Adam")
plt.plot(history_dense_1.history['val_categorical_accuracy'], label = "test Adam")


plt.legend()
plt.show()

### with two layers


history_dense_2 = History()
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(64, activation="relu"))
model.add(Dense(n_classes,activation="softmax"))
model.summary()

early_stopping = EarlyStopping(patience=20,monitor="val_loss")
model.compile(loss="categorical_crossentropy",optimizer="adam",
              metrics=["categorical_accuracy"])
model.fit(X_train, y_train, validation_split=0.25,epochs=100,
          callbacks=[early_stopping, history_dense_2])

print(model.evaluate(X_test,y_test))

plt.plot(history_dense_2.history['categorical_accuracy'], label = "train Adam")
plt.plot(history_dense_2.history['val_categorical_accuracy'], label = "test Adam")


plt.legend()
plt.show()

### With convolution

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

history_conv_1 = History()
model = Sequential()
model.add(Conv2D(16,(3,3),input_shape=X_train.shape[1:]))
model.add(Flatten())
model.add(Dense(n_classes,activation="softmax"))
model.summary()

early_stopping = EarlyStopping(patience=30,monitor="val_loss")
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["categorical_accuracy"])
model.fit(X_train, y_train, validation_split=0.25,epochs=100,callbacks=[early_stopping, history_conv_1])
model.evaluate(X_test,y_test)

plt.plot(history_dense_1.history['categorical_accuracy'], label = "tarina dense Layer=1")
plt.plot(history_dense_1.history['val_categorical_accuracy'], label = "test dense Layer=1")

# plt.plot(history_dense_2.history['categorical_accuracy'], label = "tarina dense Layer=2")
# plt.plot(history_dense_2.history['val_categorical_accuracy'], label = "test dense Layer=2")

plt.plot(history_conv_1.history['categorical_accuracy'], label = "tarina Conv Layer=1")
plt.plot(history_conv_1.history['val_categorical_accuracy'], label = "test Conv Layer=1")

plt.legend()
plt.show()


history_conv_max_1 = History()
model = Sequential()
model.add(Conv2D(16,(3,3),input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(n_classes,activation="softmax"))
model.summary()

early_stopping = EarlyStopping(patience=30,monitor="val_loss")
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["categorical_accuracy"])
model.fit(X_train, y_train, validation_split=0.25,epochs=100,callbacks=[early_stopping, history_conv_max_1])
model.evaluate(X_test,y_test)


plt.plot(history_dense_1.history['categorical_accuracy'], label = "tarina dense Layer=1")
plt.plot(history_dense_1.history['val_categorical_accuracy'], label = "test dense Layer=1")

# plt.plot(history_dense_2.history['categorical_accuracy'], label = "tarina dense Layer=2")
# plt.plot(history_dense_2.history['val_categorical_accuracy'], label = "test dense Layer=2")

plt.plot(history_conv_1.history['categorical_accuracy'], label = "tarina Conv Layer=1")
plt.plot(history_conv_1.history['val_categorical_accuracy'], label = "test Conv Layer=1")

plt.plot(history_conv_max_1.history['categorical_accuracy'], label = "tarina Conv Max Layer=1")
plt.plot(history_conv_max_1.history['val_categorical_accuracy'], label = "test Conv Max Layer=1")

plt.legend()
plt.show()

#### with same padding

history_conv_max_2 = History()
model = Sequential()
model.add(Conv2D(16,(3,3), padding="same",
                 input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(n_classes,activation="softmax"))
model.summary()

early_stopping = EarlyStopping(patience=30,monitor="val_loss")
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["categorical_accuracy"])
model.fit(X_train, y_train, validation_split=0.25,epochs=100,callbacks=[early_stopping, history_conv_max_2])
model.evaluate(X_test,y_test)


plt.plot(history_dense_1.history['categorical_accuracy'], label = "tarina dense Layer=1")
plt.plot(history_dense_1.history['val_categorical_accuracy'], label = "test dense Layer=1")

# plt.plot(history_dense_2.history['categorical_accuracy'], label = "tarina dense Layer=2")
# plt.plot(history_dense_2.history['val_categorical_accuracy'], label = "test dense Layer=2")

plt.plot(history_conv_1.history['categorical_accuracy'], label = "tarina Conv Layer=1")
plt.plot(history_conv_1.history['val_categorical_accuracy'], label = "test Conv Layer=1")

plt.plot(history_conv_max_2.history['categorical_accuracy'], label = "tarina Conv Max Layer=1")
plt.plot(history_conv_max_2.history['val_categorical_accuracy'], label = "test Conv Max Layer=1")

plt.legend()
plt.show()


#### with 2 layers

history_conv_max_3 = History()
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=X_train.shape[1:],padding="same"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(16,(2,2),padding="same"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(n_classes,activation="softmax"))
model.summary()

early_stopping = EarlyStopping(patience=30,monitor="val_loss")
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["categorical_accuracy"])
model.fit(X_train, y_train, validation_split=0.25,epochs=100,callbacks=[early_stopping, history_conv_max_3])
model.evaluate(X_test,y_test)

# plt.plot(history_dense_1.history['categorical_accuracy'], label = "tarina dense Layer=1")
# plt.plot(history_dense_1.history['val_categorical_accuracy'], label = "test dense Layer=1")

plt.plot(history_dense_2.history['categorical_accuracy'], label = "tarina dense Layer=2")
plt.plot(history_dense_2.history['val_categorical_accuracy'], label = "test dense Layer=2")

# plt.plot(history_conv_max_1.history['categorical_accuracy'], label = "tarina Conv Layer=1")
# plt.plot(history_conv_max_1.history['val_categorical_accuracy'], label = "test Conv Layer=1")

plt.plot(history_conv_max_3.history['categorical_accuracy'], label = "tarina Conv Layer=2")
plt.plot(history_conv_max_3.history['val_categorical_accuracy'], label = "test Conv Layer=2")

plt.legend()
plt.show()


#### with batch normalization

history_conv_max_4 = History()
model = Sequential()
model.add(Conv2D(64, (3, 3),
                 input_shape=X_train.shape[1:],padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(n_classes,activation="softmax"))
model.summary()

early_stopping = EarlyStopping(patience=30,monitor="val_loss")
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["categorical_accuracy"])
model.fit(X_train, y_train, validation_split=0.25,epochs=100,callbacks=[early_stopping, history_conv_max_4])
model.evaluate(X_test,y_test)


plt.plot(history_dense_2.history['categorical_accuracy'], label = "tarina dense Layer=2")
plt.plot(history_dense_2.history['val_categorical_accuracy'], label = "test dense Layer=2")


plt.plot(history_conv_max_4.history['categorical_accuracy'], label = "Batch normalization")
plt.plot(history_conv_max_4.history['val_categorical_accuracy'], label = "Batch normalization")

plt.legend()
plt.show()