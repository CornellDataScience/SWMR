import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Model1:

    def __init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate):

        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(size_layer, activation = tf.nn.relu)

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

        # [dimension of word, batch size, dimension input]
        self.X = tf.placeholder(tf.float32, [None, None, dimension_input])

        #[batch size, dimension input]
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        self.outputs, self.last_state = tf.nn.dynamic_rnn(self.rnn_cells, self.X, dtype = tf.float32)

        self.rnn_W = tf.Variable(tf.random_normal((size_layer, dimension_output)))
        self.rnn_B = tf.Variable(tf.random_normal([dimension_output]))

        self.logits = tf.matmul(self.outputs[-1], self.rnn_W) + self.rnn_B

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.Y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
