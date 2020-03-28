# Image Classification

# Importing the libraries
from __future__ import print_function, division
from builtins import range
import os
from keras.layers import Input, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Lambda, Concatenate, Dense
from keras.models import Model
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_mnist(limit = None):
    if not os.path.exists('/Users/admin/Desktop/Udemy/Advanced deep learning/digit-recognizer'):
        print('You must create a folder called Jigsaw')
    if not os.path.exists('/Users/admin/Desktop/Udemy/Advanced deep learning/digit-recognizer/train.csv'):
        print("Looks like you haven't downloaded the data or its not in the right spot")
        
    print("Reading in and transformin data")
    df = pd.read_csv('/Users/admin/Desktop/Udemy/Advanced deep learning/digit-recognizer/train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:,1:].reshape(-1,28,28)/255.0
    Y = data[:,0]
    if limit is not None:
        X,Y = X[:limit],Y[:limit]
    return X,Y


# Get the data
X, Y = get_mnist()

# Configuration
D = 28
M = 15

# Input an image of size 28 X 28
input_ = Input(shape = (D,D))

# Up-down  
rnn1 = Bidirectional(LSTM(M, return_sequences = True)) 
x1 = rnn1(input_)
x1 = GlobalMaxPooling1D()(x1)

# Left-right
rnn2 = Bidirectional(LSTM(M, return_sequences = True)) 

# Custom layer
permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0,2,1)))
x2 = permutor(input_)
x2 = rnn2(x2)                  # output is N X D X 2M
x2 = GlobalMaxPooling1D()(x2)  # output is N X 2M


# Put them together
concatenator = Concatenate(axis = 1)
x = concatenator([x1,x2])      # output is N X 4M


# Final dense layer
output = Dense(10, activation = 'softmax')(x)
model = Model(inputs = input_, outputs = output)


# Compile
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'])


# Train
print('Training model...')
r = model.fit(X,Y, batch_size = 32, epochs = 10, validation_split = 0.3)

# Plot some data
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show


# Accuracies
plt.plot(r.history['acc'], label = 'acc')
plt.plot(r.history['val_acc'], label = 'val_acc')
plt.legend()
plt.show









   