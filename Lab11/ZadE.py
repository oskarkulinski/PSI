import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras

from numpy.random import seed
seed(123)


sgd1 = keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)

sgd2 = keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

RMSprop = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

Adagrad = keras.optimizers.Adagrad(learning_rate=0.01)

Adadelta = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)

Adam1 = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

Adam2 = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

optimizers = {'sgd' : sgd1, 'sgd nesterov': sgd2, 'RMSprop':RMSprop,
             'Adagrad':Adagrad, 'Adadelta':Adadelta,
             'Adam LR 0.001': Adam1, 'Adam LR 0.0001': Adam2}

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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense

from keras.callbacks import History
for optimizer in optimizers:
    model = Sequential()
    model.add(Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)))
    model.add(Dense(50,activation="sigmoid"))
    model.add(Dense(10,activation="sigmoid"))
    model.add(Dense(1,activation="sigmoid"))
    model.summary()

    model.compile(loss="binary_crossentropy",optimizer=optimizers[optimizer],
                            metrics=["accuracy"])
    history = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32,epochs=100)

    plt.plot(history.history['accuracy'], label = "train")
    plt.plot(history.history['val_accuracy'], label = "test")
    plt.title(optimizer + ' accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label = "train")
    plt.plot(history.history['val_loss'], label = "test")
    plt.title(optimizer + ' loss')
    plt.legend()
    plt.show()