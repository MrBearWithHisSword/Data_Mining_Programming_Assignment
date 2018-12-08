import tensorflow as tf
import numpy as np
from hyperparams import  Hyperparams as hp


def normalize(inputs,
              epsilon=1e-8,
              scope='layer_norm',
              reuse=None):
    """Applies layer normalization.

    :param inputs: A tensor with first dimension as 'batch_size'
    :param epsilon: preventing ZeroDivision Error
    :param scope: tf.variable_scope
    :param reuse: Boolean, whether to reuse variables

    :return: A normalized tensor with the same shape and dtype as 'inputs'

    note: 考虑到这里的inputs有可能是padding后的词向量，因此在batch_size这一维度上对
    做normalization是不合理的，较为合理的方式是对inputs的最后一维做norm.
    """

    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        # 用这种方式将会导致beta和gamma的值无法更新
        # beta = tf.Variable(tf.zeros(params_shape))
        # gamma = tf.Variable(tf.ones(params_shape))
        beta = tf.get_variable('beta', shape=params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', shape=params_shape, initializer=tf.zeros_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma*normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              trainable=True,
              lookup_table_source=None,
              scope='embedding',
              reuse=None):
    """

    :param inputs:
    :param vocab_size:
    :param num_units:
    :param zero_pad:
    :param scale:
    :param trainable: Boolean, whether trainable.
    :param lookup_table_source: Your own word_embedding matrix.
    :param scope: tf.variable_scope
    :param reuse: whether to reuse variable

    :return: embedded inputs
    """
    with tf.variable_scope(scope, reuse=reuse):
        # 不导入外部词向量
        if lookup_table_source is None:
            lookup_table = tf.get_variable('lookup_table',
                                           trainable=trainable,
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())
        # 导入外部词向量
        else:
            lookup_table = tf.to_float(lookup_table_source)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table), axis=0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def positional_enconding(inputs,
                         num_units,
                         zero_pad=True,
                         strategy='sinusoidal',
                         vocab_size=None,
                         scale=True,
                         scope='positional_encoding',
                         reuse=None):
    """Applies positional_encoding

    :param inputs: A tensor with shape of (batch_size, T)
    :param num_units:
    :param zero_pad:
    :param strategy:
    :param vocab_size:
    :param scale:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        if strategy == 'sinusoidal':
            # if
            batch_size, T = inputs.get_shape().as_list()
            # position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [batch_size, 1])
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [1, 1])
            position_enc = np.array([
                [pos/(np.power(10000, 2*i/num_units)) for i in range(num_units)]
                for pos in range(T)])
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
            lookup_table = tf.convert_to_tensor(position_enc)

            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=([1, num_units])), lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
            if scale:
                outputs = outputs * (num_units ** 0.5)
        elif strategy == 'learn':
            outputs = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1]),
                                vocab_size=vocab_size,
                                num_units=num_units,
                                zero_pad=zero_pad,
                                scale=scale,
                                scope=scope)
        else:
            raise RuntimeError('on such type of positional encoding strategy.')
        return tf.to_float(outputs)


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=True,
                        scope='multihead_attention',
                        reuse=None):
    """

    :param queries:
    :param keys:
    :param num_units:
    :param num_heads:
    :param dropout_rate:
    :param is_training:  whether to apply dropout.
    :param causality:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)

        # Split and concat(switch to  multihead)
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Matmul
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))    # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (num_units ** 0.5)

        '''
        Masking:
        在做self-attention时Queries和Keys的信息是同步流入的,因此他们
        可以使用相通的masks, 但对于nonself-attention,Queries和Keys
        可能是不同步的，因此需要区分query_masks和key_masks
        '''
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))   # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Causality(Future blinding)
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        # Activation
        outputs = tf.nn.softmax(outputs)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        # print(tf.reduce_sum(queries))
        # print(query_masks)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)  # (h*N, T_q, T_k)

        # compatibility * Value(Key)
        outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalization
        outputs = normalize(outputs)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope='feedforward',
                reuse=None):
    """Point-wise feed_forward net.

    :param inputs: 3D tensor with shape of [N, T, C]
    :param num_units:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {'inputs': inputs, 'filters': num_units[0], 'kernel_size': 1,
                  'activation': tf.nn.relu, 'use_bias': True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {'inputs': outputs, 'filters': num_units[1], 'kernel_size': 1,
                  'activation': None, 'use_bias': True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalization
        outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    """Applies label smoothing. paper from: https://arxiv.org/abs/1512.00567.

    :param inputs: 3D tensor with shape of [N, T, V], where V is the num of classes.
    :param epsilon:
    :return:
    """

    num_channel = inputs.get_shape().as_list()[-1]
    return (1-epsilon)*tf.to_float(inputs) + (epsilon / num_channel)
