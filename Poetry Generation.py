# Poetry generation

# Importing the libraries
from __future__ import print_function, division
from builtins import range
import os
import sys
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,LSTM,Embedding
from keras.optimizers import Adam,SGD
from keras.models import Model


# Some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 2000
LATENT_DIM = 25


# Load data
input_texts = []
target_texts = []
for line in open('/Users/admin/Desktop/Udemy/Advanced deep learning/machine_learning_examples-master/hmm_class/robert_frost.txt'):
    line = line.rstrip()
    if not line:
        continue
    input_line = '<sos>' + line
    target_line = line + '<eos>'
    input_texts.append(input_line)
    target_texts.append(target_line)
    
all_lines = input_line + target_line


# Convert the sentences into integer
tokenizers = Tokenizer(num_words = MAX_VOCAB_SIZE, filters = '')
tokenizers.fit_on_texts(all_lines)
input_sequences = tokenizers.texts_to_sequences(input_texts)
target_sequences = tokenizers.texts_to_sequences(target_texts)


# Find max sequence length
max_sequence_length_from_data = max(len(s) for s in input_sequences)
print('Max sequence length:', max_sequence_length_from_data)


# Get word -> integer mapping
word2idx = tokenizers.word_index
print('Found %s unique tokens' %len(word2idx))
assert('<sos>' in word2idx)
assert('<eos>' in word2idx)

# Pad sequences so that we get a N X T matrix
max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_length, padding = 'post')
target_sequences = pad_sequences(target_sequences, maxlen = max_sequence_length, padding = 'post')
print('Shape of data tensor:', input_sequences.shape)



# Load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('/Users/admin/Desktop/Udemy/Advanced deep learning/glove/glove.6B.%sd.txt'% EMBEDDING_DIM)) as f:
    # is a space separated text file in the format word vec[0] vec[1] vec[2]...
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype = 'float32')
        word2vec[word] = vec
print('Found %s word vectors' % len(word2vec))


# Prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word,i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# One hot the targets (can't use sparse entropy)
one_hot_targets = np.zeros((len(input_sequences),max_sequence_length,num_words))
for i, target_sequences in enumerate(target_sequences):
    for t, word in enumerate(target_sequences):
        if word>0:
            one_hot_targets[i,t,word] = 1
            
            
# Load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(
        num_words,
        EMBEDDING_DIM,
        weights = [embedding_matrix]
        )


# Load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights = [embedding_matrix])

print('Building Model...')
# Create an LSTM network with a single lstm
input_ = Input(shape = (MAX_SEQUENCE_LENGTH,))
input_h= Input(shape = (LATENT_DIM,))
input_c = Input(shape = (LATENT_DIM,))
x = embedding_layer(input_)
lstm = LSTM(LATENT_DIM,return_sequences = True, return_state = True)
x,_,_ = lstm(x,initial_state = [input_h,input_c])
dense = Dense(num_words, activation = 'softmax')
output = dense(x)


# Compile the model for training
model = Model([input_,input_h,input_c], output)
model.compile(
        loss = 'categorical_crossentropy',
        optimizer = Adam(lr =0.01),
        metrics = ['accuracy']
        )

print('Training model...')
z = np.zeros((input_sequences),LATENT_DIM)
r = model.fit(
        [input_sequences,z,z],
        one_hot_targets,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        validation_split = VALIDATION_SPLIT
        )


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








