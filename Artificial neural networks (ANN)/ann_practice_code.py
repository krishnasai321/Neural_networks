# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:22:40 2021

@author: Sai Krishna
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import keras
import os

os.chdir('C:\\Data\github\\neural_networks\\Artificial neural networks (ANN)')

df = pd.read_csv('sample_input.csv')

df = df.drop(['hcp_id'],axis = 1)

df = pd.get_dummies(df,drop_first=True)

df = df.values

X = df[:,0:29]
Y = df[:,29]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#Keras - Neural Network portion
#Adding dense layers, various activation functions, regularizers, drop outs, multiple metrics, validation split etc.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers

model = Sequential()
model.add(Dense(16, input_dim=29, activation='relu'))
model.add(Dense(8, activation='relu',kernel_regularizer = (regularizers.l1_l2(l1=0.01,l2=0.01))))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision','Recall'])
#model.optimizer.lr = 0.01

nn = model.fit(X, Y, epochs=150, batch_size=50,verbose=0)
#nn = model.fit(X, Y, epochs=150, batch_size=50, validation_split= 0.2)

plt.plot(nn.history['precision'])
plt.show()
