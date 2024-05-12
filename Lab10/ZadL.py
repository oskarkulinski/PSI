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

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

print(model.evaluate(X_test, y_test))

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict(X_new)
predicted_classes = np.argmax(y_pred, axis=1)

model_elu = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(10, activation="softmax")
])

model_elu.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history_elu = model_elu.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))


model_leaky = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="leaky_relu"),
    keras.layers.Dense(100, activation="leaky_relu"),
    keras.layers.Dense(10, activation="softmax")
])


model_leaky.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history_leaky = model_leaky.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

plt.figure(figsize=(8, 5))
plt.plot(history_elu.history['loss'], label='Elu loss')
plt.plot(history_elu.history['accuracy'], label='Elu accuracy')
plt.plot(history_leaky.history['loss'], label='Leaky loss')
plt.plot(history_leaky.history['accuracy'], label='Leaky accuracy')
plt.plot(history.history['loss'], label='Relu loss')
plt.plot(history.history['accuracy'], label='Relu accuracy')
plt.grid(True)
plt.legend()
plt.gca().set_ylim(0, 1)
plt.show()