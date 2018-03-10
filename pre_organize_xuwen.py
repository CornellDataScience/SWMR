import collections
import random
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class WordModel:

    def __init__(self, batch_size, dimension_size, learning_rate, vocabulary_size):

        self.train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

        # randomly generated initial value for each word dimension, between -1.0 to 1.0
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, dimension_size], -1.0, 1.0))

        # find train_inputs from embeddings
        embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

        # estimation for not normalized dataset
        self.nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, dimension_size], stddev = 1.0 / np.sqrt(dimension_size)))

        # each node have their own bias
        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # calculate loss from nce, then calculate mean
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = self.nce_weights, biases = self.nce_biases, labels = self.train_labels,
                                                  inputs = embed, num_sampled = batch_size / 2, num_classes = vocabulary_size))

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # normalize the data by simply reduce sum
        self.norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

        # normalizing each embed
        self.normalized_embeddings = embeddings / self.norm

def read_data(filename):
    '''input filename
        return string: all the text (string); concated: list of string;
        label; list(set(string)): all the unique words
    '''
    dataset = pd.read_csv(filename)
    rows = dataset.shape[0]
    print('there are', rows, 'total rows')

    # last column is our target
    label = dataset.ix[:, 2].values

    # get second and third column values
    concated = []
    data = dataset.ix[:, 1].values

    for i in range(data.shape[0]):
        string = ""

#         for k in range(data.shape[1]):
        string += data[i] + " "

        concated.append(string)

    # get all split strings from second column and second last column
    dataset = dataset.ix[:, 1].values
    string = []
    for i in range(dataset.shape[0]):
#         for k in range(dataset.shape[1]):
        string += dataset[i].split()

    return string, concated, label, list(set(string))

def build_dataset(words, vocabulary_size):
    count = []
    # extend count
    # sorted decending order of words
    count.extend(collections.Counter(words).most_common(vocabulary_size))

    dictionary = dict()
    for word, _ in count:
        #simply add dictionary of word, used frequently placed top
        dictionary[word] = len(dictionary)

    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
# data: index of words
        data.append(index)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, dictionary, reverse_dictionary

def generate_batch_skipgram(words, batch_size, num_skips, skip_window):
    data_index = 0

    #check batch_size able to convert into number of skip in skip-grams method
    assert batch_size % num_skips == 0

    assert num_skips <= 2 * skip_window

    # create batch for model input
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1

    # a buffer to placed skip-grams sentence
    buffer = collections.deque(maxlen=span)

    for i in range(span):
        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]

        for j in range(num_skips):

            while target in targets_to_avoid:
                # random a word from the sentence
                # if random word still a word already chosen, simply keep looping
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)

    data_index = (data_index + len(words) - span) % len(words)
    return batch, labels

def generatevector(dimension, batch_size, skip_size, skip_window, num_skips, iteration, words_real):

    print('data size: ', len(words_real))
    data, dictionary, reverse_dictionary = build_dataset(words_real, len(words_real))

    sess = tf.InteractiveSession()
    print('Creating Word2Vec model.')

    model = WordModel(batch_size, dimension, 0.01, len(dictionary))
    sess.run(tf.global_variables_initializer())

    last_time = time.time()

    for step in range(iteration):
        new_time = time.time()
        batch_inputs, batch_labels = generate_batch_skipgram(data, batch_size, num_skips, skip_window)
        feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}

        _, loss = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)

        if ((step + 1) % 1000) == 0:
            print('epoch: ', step + 1, ', loss: ', loss, ', speed: ', time.time() - new_time)

    tf.reset_default_graph()
    return dictionary, reverse_dictionary, model.normalized_embeddings.eval()
