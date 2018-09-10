import tensorflow as tf
from tensorflow.contrib import rnn


class RecNet(object):
    def __init__(self, sess, conf, model):
        self.sess = sess
        self.conf = conf
        self.feature_extractor = model(tf.Session(), conf)
        self.weights = weight_variable(shape=[self.conf.num_hidden_units, self.conf.num_cls])
        self.biases = bias_variable(shape=[self.conf.num_cls])
        self.y = tf.placeholder(tf.float32, [None, self.conf.max_time, self.conf.num_cls])
        self.seqLen = tf.placeholder(tf.int32, [None])

    def build(self):
        features = self.feature_extractor.features
        num_features = features.get_shape().as_list()[-1]
        x = tf.reshape(features, [-1, self.conf.max_time, num_features])
        if self.conf.recurrent_model == 'RNN':
            cell = rnn.BasicRNNCell(self.conf.num_hidden)
            outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=self.seqLen, dtype=tf.float32)
        elif self.conf.recurrent_model == 'LSTM':
            cell = rnn.BasicLSTMCell(self.conf.num_hidden)
            outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=self.seqLen, dtype=tf.float32)
        num_examples = tf.shape(x)[0]
        w_repeated = tf.tile(tf.expand_dims(self.weights, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, w_repeated) + self.biases
        out = tf.squeeze(out)
        return out


# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initer = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initer)