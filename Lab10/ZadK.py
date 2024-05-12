import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Wczytaj dane treningowe i testowe
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


from keras.models import Sequential
from keras.layers import Dense

from keras.callbacks import History

history = History()
model = Sequential()
model.add(Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])

# history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')

# Reshape labels if necessary (for example, for categorical crossentropy)
y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)
history = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32,epochs=100)


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


print(model.evaluate(X_test,y_test))

from sklearn.metrics import accuracy_score

# Get predictions
y_pred = model.predict(X_test)

# Assuming a binary classification problem
y_pred = (y_pred > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model_sigmoid = Sequential([
    Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)),
    Dense(50,activation="sigmoid"),
    Dense(10,activation="sigmoid"),
    Dense(1,activation="sigmoid")
])

model_sigmoid.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])

model_tanh = Sequential([
    Dense(100,activation="tanh",input_shape=(X_train.shape[1],)),
    Dense(50,activation="tanh"),
    Dense(10,activation="tanh"),
    Dense(1,activation="tanh")
])

model_tanh.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])

model_relu = Sequential([
    Dense(100,activation="relu",input_shape=(X_train.shape[1],)),
    Dense(50,activation="relu"),
    Dense(10,activation="relu"),
    Dense(1,activation="relu")
])

model_relu.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])

model_elu = Sequential([
    Dense(100,activation="elu",input_shape=(X_train.shape[1],)),
    Dense(50,activation="elu"),
    Dense(10,activation="elu"),
    Dense(1,activation="elu")
])

model_elu.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])

model_leakyrelu = Sequential([
    Dense(100,activation="leaky_relu",input_shape=(X_train.shape[1],)),
    Dense(50,activation="leaky_relu"),
    Dense(10,activation="leaky_relu"),
    Dense(1,activation="leaky_relu")
])

model_leakyrelu.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])

history_sigmoid = model_sigmoid.fit(X_train_scaled,y_train,epochs=30)
history_tanh = model_tanh.fit(X_train_scaled,y_train,epochs=30)
history_relu = model_relu.fit(X_train_scaled,y_train,epochs=30)
history_elu = model_elu.fit(X_train_scaled,y_train,epochs=30)
history_leaky = model_leakyrelu.fit(X_train_scaled,y_train,epochs=30)

plt.plot(history_sigmoid.history['loss'], label='Sigmoid Loss')
plt.plot(history_sigmoid.history['accuracy'], label='Sigmoid Accuracy')
plt.plot(history_tanh.history['loss'], label='Tanh Loss')
plt.plot(history_tanh.history['accuracy'], label='Tanh Accuracy')
plt.plot(history_leaky.history['loss'], label='Leaky Loss')
plt.plot(history_leaky.history['accuracy'], label='Leaky Accuracy')
plt.plot(history_elu.history['loss'], label='Elu Loss')
plt.plot(history_elu.history['accuracy'], label='Elu Accuracy')
plt.plot(history_relu.history['loss'], label='Relu Loss')
plt.plot(history_relu.history['accuracy'], label='Relu Accuracy')
plt.legend()
plt.show()

