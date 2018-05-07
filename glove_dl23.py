from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import os

def get_glove_m(docs):
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    # pad documents to a max length of 4 words
    max_length = 3000
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(os.path.dirname(os.getcwd())+'/models/glove.6B.100d.txt')


    for line in f:
    	values = line.split()
    	word = values[0]

    	coefs = asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs

    f.close()
    # print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return vocab_size, embedding_matrix, padded_docs

def padded_doc(docs):
    t = Tokenizer()
    t.fit_on_texts(docs)
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    # pad documents to a max length of 4 words
    max_length = 3000
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # print(padded_docs)
    return padded_docs
