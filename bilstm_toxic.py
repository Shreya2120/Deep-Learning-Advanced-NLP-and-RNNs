# Keras Example

# Importing the libraries
from __future__ import print_function, division
from builtins import range
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,GlobalMaxPooling1D
from keras.layers import LSTM,Bidirectional,Embedding,Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score


# Some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10


# Download the data:
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Download the word vectors:
# http://nlp.stanford.edu/data/glove.6B.zip


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


# Prepare text samples and their arrays
train = pd.read_csv('/Users/admin/Desktop/Udemy/Advanced deep learning/jigsaw-toxic-comment-classification-challenge/train.csv')
sentences = train['comment_text'].fillna('DUMMY_VALUE').values
possible_labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
targets = train[possible_labels].values
print('max sequence length', max(len(s) for s in sentences))
print('min sequence length', min(len(s) for s in sentences))
s = sorted(len(s) for s in sentences)
print('median sequence length', s[len(s)//2])


# Convert the sentences to integers
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)


# Get word to integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens',word2idx)


# Pad sequences so that we get a N X T matrix
data = pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:',data.shape)


# Prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word,i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
            
# Load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(
        num_words,
        EMBEDDING_DIM,
        weights = [embedding_matrix],
        input_length = MAX_SEQUENCE_LENGTH,
        trainable = False
        )


# Train a 1D convent with global maxpooling
print('Building Model...')
input_ = Input(shape = (MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Bidirectional(LSTM(15, return_sequences = True))(x)
x = GlobalMaxPooling1D()(x)
output = Dense(len(possible_labels), activation = 'sigmoid')(x)


# Compile the model for training
model = Model(input_, output)
model.compile(
        loss = 'binary_crossentropy',
        optimizer = Adam(lr = 0.01),
        metrics = ['accuracy']
        )

print('Training model...')
r = model.fit(
        data,
        targets,
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


# Plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j],
                        p[:,j])
    aucs.append(auc)
print(np.mean(aucs))







