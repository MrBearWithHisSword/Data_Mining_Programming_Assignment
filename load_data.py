# -*- coding: utf-8 -*-
import pickle as pk
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from hyperparams import Hyperparams as hp


def read_bunch_obj(path):

    with open(path, "rb") as file_obj:
        bunch = pk.load(file_obj)
    return bunch


def load_train_data():

    data_dir = hp.train_data_dir
    maxlen = hp.sentence_len

    bunch = read_bunch_obj(data_dir)

    # construct word2index
    sentences = [item.strip().split(" ") for item in bunch.content]
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(sentences)
    word2index = tokenizer.word_index
    index2word = tokenizer.index_word



    # padding inputs
    X = tokenizer.texts_to_sequences(sentences)
    for i in range(len(X)):
        X[i] = np.pad(np.array(X[i]), (0, maxlen - len(X[i])), 'constant')
    X = np.array(X)
    Y = np.array(bunch.label).reshape(len(bunch.label), -1)
    Y = to_categorical(np.asarray(Y))
    # print(X.shape)
    # print(Y.shape)

    # import word_embedding
    if hp.load_embedding_matrix is True:
        embedding_matrix = np.zeros((len(index2word)+1, hp.embedding_dim))
        embeddings_index = {}
        with open('word_embedding/sgns.wiki.bigram-char', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]  # 单词
                # print(word)
                coefs = np.asarray(values[1:], dtype='float32')  # 单词对应的向量
                # print(coefs)
                embeddings_index[word] = coefs  # 单词及对应的向量
        # print(embeddings_index['玩具'])
        for word, i in word2index.items():
            embedding_vector = embeddings_index.get(word)  # 根据词向量字典获取该单词对应的词向量
            # print(embedding_vector)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        # print(embedding_matrix[1])
        return X, Y, word2index, index2word, embedding_matrix
    else:
        return X, Y, word2index, index2word, None


def get_dataset():
    np.random.seed(0)
    X, Y, word2index, index2word, embedding_matrix = load_train_data()
    shuffle_list = np.arange(len(X))
    np.random.shuffle(shuffle_list)
    X = X[shuffle_list, :]
    Y = Y[shuffle_list, :]
    X_train = X[:hp.training_examples, :]
    Y_train = Y[:hp.training_examples, :]
    hp.X_test = X[hp.training_examples:, :]
    hp.Y_test = Y[hp.training_examples:, :]
    X_train = tf.convert_to_tensor(X_train)
    Y_train = tf.convert_to_tensor(Y_train)
    hp.embedding_matrix = embedding_matrix
    # print(hp.embedding_matrix[1])

    dataset = tf.data.Dataset.from_tensor_slices({'inputs':X_train, 'labels':Y_train})
    # dataset = dataset.repeat(hp.num_epochs)
    # dataset = dataset.batch(hp.batch_size)
    return dataset, word2index, index2word

