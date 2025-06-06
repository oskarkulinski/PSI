import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

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
model.summary()

model.compile(loss="binary_crossentropy",optimizer="Adam",
              metrics=["accuracy"])

from keras.callbacks import EarlyStopping
X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
#history = model.fit(X_train, y_train, validation_data= (X_test, y_test),
 #                   batch_size=32,epochs=100, callbacks=[early_stopping])

#pd.DataFrame(history.history).plot(figsize=(8, 5))
#plt.grid(True)
#plt.gca().set_ylim(0, 1)
#plt.show()

#model.evaluate(X_test,y_test)


from sklearn import  metrics
metrics.accuracy_score(y_true= y_test, y_pred= np.argmax(model.predict(X_test), axis=-1) )

#### Make moons

from sklearn.datasets import make_moons
# generate 2d classification dataset
X, y = make_moons(n_samples=100, noise=0.2, random_state=5)
# split into train and test
# n_train = 30
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=2)

n_train = 80
n_val = 10
X_train, X_test = X[:n_train, :], X[n_train:, :]
y_train, y_test = y[:n_train], y[n_train:]

X_val = X_train[:n_val, :]
X_train = X_train[n_val:, :]
y_val = y_train[:n_val]
y_train = y_train[n_val:]

plt.scatter(X_train[:,0],X_train[:,1], c=y_train)
plt.show()

model1 = keras.models.Sequential([
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model2 = keras.models.Sequential([
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5, mode='min', verbose=1)


model1.compile(loss="binary_crossentropy",
               optimizer="Adam", metrics=["accuracy"])

model2.compile(loss="binary_crossentropy", optimizer="Adam",
               metrics=["accuracy"])

model1.fit(X_train,y_train,epochs=1000,validation_data=(X_val,y_val),
           verbose=1)
model2.fit(X_train,y_train,epochs=1000,validation_data=(X_val,y_val),
           callbacks=[early_stopping], verbose=1)

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train,y_train,model1)
plt.show()
print(model1.evaluate(X_test,y_test))

plot_decision_regions(X_train,y_train,model2)
plt.show()

print(model2.evaluate(X_test,y_test))
