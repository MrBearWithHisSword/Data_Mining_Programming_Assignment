import pickle as pk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from hyperparams import Hyperparams as hp


def read_bunch_obj(path):
    with open(path, "rb") as file_obj:
        bunch = pk.load(file_obj)
    return bunch


def load_train_data():

    data_dir = hp.train_data_dir
    maxlen = hp.sentence_len

    bunch = read_bunch_obj(data_dir)

    # construct word_index
    sentences = [item.strip().split(" ") for item in bunch.content]
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    # padding inputs
    X = tokenizer.texts_to_sequences(sentences)
    for i in range(len(X)):
        X[i] = np.pad(np.array(X[i]), (0, maxlen - len(X[i])), 'constant')
    X = np.array(X)
    Y = np.array(bunch.label).reshape(len(bunch.label), -1)
    return X, Y, word_index


def get_dataset():

    X, Y, word_inde = load_train_data()
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)

    dataset = tf.data.Dataset.from_tensor_slices({'inputs':X, 'labels':Y})
    # dataset = dataset.repeat(hp.num_epochs)
    # dataset = dataset.batch(hp.batch_size)
    return dataset


