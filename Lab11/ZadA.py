import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Wczytaj dane treningowe i testowe
train_set = pd.read_csv('adult/adult.data', sep=", ",header = None)
test_set = pd.read_csv('adult/adult.test', sep=", ",skiprows = 1, header = None) # Make sure to skip a row for the test set

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
              'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels

train = train_set.replace('?', np.nan).dropna()
test = test_set.replace('?', np.nan).dropna()

train_set.head()


dataset = pd.concat([train,test])

dataset['wage_class'] = dataset.wage_class.replace({'<=50K.': 0,'<=50K':0, '>50K.':1, '>50K':1})

dataset.drop(["fnlwgt"],axis=1,inplace=True)

dataset.drop(["education"],axis=1,inplace=True)

x = dataset.groupby('native_country')["wage_class"].mean()

d = dict(pd.cut(x[x.index!=" United-States"],5,labels=range(5)))

dataset['native_country'] = dataset['native_country'].replace(d)

dataset = pd.get_dummies(dataset,drop_first=True)

train = dataset.iloc[:train.shape[0]]
test = dataset.iloc[train.shape[0]:]

X_train = train.drop("wage_class",axis=1)
y_train = train.wage_class

X_test = test.drop("wage_class",axis=1)
y_test = test.wage_class

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

from keras.models import Sequential
from keras.layers import Dense

from keras.callbacks import History
'''
history = History()
model = Sequential()
model.add(Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])

from keras.callbacks import TensorBoard
X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

tensorboard_cb = TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
    validation_data=(X_test, y_test), callbacks=[tensorboard_cb])

'''
### MNIST
from keras.callbacks import TensorBoard
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_val = x_train[:5000]/255.0
y_val = y_train[:5000]
x_train = x_train[5000:]/255.0
y_train = y_train[5000:]
x_test = x_test/255.0
y_test = y_test

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
y_val = np_utils.to_categorical(y_val, num_classes=10)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='elu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='elu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss="categorical_crossentropy",optimizer="adam",
              metrics=["accuracy"])


tensorboard_cb = TensorBoard(run_logdir)
history = model.fit(x_train, y_train, epochs=30,
                    validation_data=(x_val, y_val), callbacks=[tensorboard_cb])

print(model.evaluate(x_test, y_test))

