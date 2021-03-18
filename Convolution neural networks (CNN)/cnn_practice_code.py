# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:10:54 2021

@author: kqll413
"""

#Import data, libraries
import keras
from keras.datasets import fashion_mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

#Loading train and test data from fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

#Reshaping for all the grayscale datasets
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)

#Changing to decimals and normalizing 0-1
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255

#Encoding dependent variable (categorical crossentropy loss)
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

#Setting up NN layers
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10))
model.add(Activation('softmax'))

#Compiling NN
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#Fitting the models
model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=10)

#Prediction
predictions = model.predict(test_X)
print(np.argmax(np.round(predictions[0])))

#Plot first image
plt.imshow(test_X[0].reshape(28, 28), cmap = plt.cm.binary)
plt.show()