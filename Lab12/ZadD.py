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

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model_fc = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model_cnn = keras.models.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation="softmax")
])

model_fc.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


history_fc = model_fc.fit(X_train_full, y_train_full,
                          validation_split=0.2, epochs=30)
history_cnn = model_cnn.fit(X_train_full, y_train_full,
                            validation_split=0.2, epochs=30)

print(model_fc.evaluate(X_test, y_test))
print(model_cnn.evaluate(X_test, y_test))

plt.plot(history_fc.history['accuracy'], label="train FC")
plt.plot(history_fc.history['val_accuracy'], label="validation FC")
plt.plot(history_cnn.history['accuracy'], label="train CNN")
plt.plot(history_cnn.history['val_accuracy'], label="validation CNN")

plt.legend()
plt.show()

