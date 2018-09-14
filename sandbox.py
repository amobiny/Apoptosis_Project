import tensorflow as tf
from tensorflow.contrib import rnn
from DataLoaders.Feature_ApoptosisLoader import DataLoader
from config import args as conf
import numpy as np
from sklearn.metrics import confusion_matrix
import os

num_hidden = 200
max_time = 72
num_cls = 2
batch_size = 4
input_dim = 512
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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


x = tf.placeholder(tf.float32, [batch_size, max_time, input_dim])
y = tf.placeholder(tf.float32, [batch_size, max_time, num_cls])
seqLen = tf.placeholder(tf.int32, [batch_size])

with tf.variable_scope('RecNet'):
    weights = weight_variable(shape=[num_hidden, num_cls])
    biases = bias_variable(shape=[num_cls])

cell = rnn.BasicLSTMCell(num_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seqLen, dtype=tf.float32)
batch_s = tf.shape(x)[0]
w_repeated = tf.tile(tf.expand_dims(weights, 0), [batch_s, 1, 1])
logits = tf.reshape(tf.matmul(outputs, w_repeated) + biases, [batch_size * max_time, num_cls])
y_pred_tensor = tf.argmax(logits, axis=-1, name='predictions')
labels = tf.reshape(y, [batch_size * max_time, num_cls])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                 logits=logits), name='loss')
# loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y,
#                                                                logits=logits,
#                                                                pos_weight=3), name='loss')
mean_loss, mean_loss_op = tf.metrics.mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam-op')
train_vars = tf.trainable_variables()
train_op = optimizer.minimize(loss, var_list=train_vars)

correct_prediction = tf.equal(y_pred_tensor, tf.argmax(labels, -1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
mean_accuracy, mean_accuracy_op = tf.metrics.mean(accuracy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    best_validation_accuracy = 0
    data_reader = DataLoader(conf)
    data_reader.get_data(mode='train')
    data_reader.get_data(mode='valid')
    num_train_batch = data_reader.count_num_batch(batch_size, mode='train')
    num_val_batch = data_reader.count_num_batch(batch_size, mode='valid')
    for epoch in range(1000):
        data_reader.randomize()
        for train_step in range(num_train_batch):
            glob_step = epoch * num_train_batch + train_step
            start = train_step * batch_size
            end = (train_step + 1) * batch_size
            x_batch, y_batch = data_reader.next_batch(start, end, mode='train')
            feed_dict = {x: x_batch, y: y_batch, seqLen: max_time * np.ones(batch_size)}
            if train_step % conf.SUMMARY_FREQ == 0:
                _, _, _ = sess.run([train_op,
                                             mean_loss_op,
                                             mean_accuracy_op],
                                            # merged_summary],
                                            feed_dict=feed_dict)
                loss, acc = sess.run([mean_loss, mean_accuracy])
                # save_summary(summary, glob_step + conf.reload_step, mode='train')
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
            else:
                sess.run([train_op, mean_loss_op, mean_accuracy_op], feed_dict=feed_dict)
        sess.run(tf.local_variables_initializer())
        y_pred = np.zeros((data_reader.y_valid.shape[0]) * conf.max_time)
        for step in range(num_val_batch):
            start = step * batch_size
            end = (step + 1) * batch_size
            x_val, y_val = data_reader.next_batch(start, end, mode='valid')
            feed_dict = {x: x_val, y: y_val,
                         seqLen: max_time * np.ones(batch_size)}
            yp, _, _ = sess.run([y_pred_tensor, mean_loss_op, mean_accuracy_op], feed_dict=feed_dict)
            y_pred[start * max_time:end * max_time] = yp
        # summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = sess.run([mean_loss, mean_accuracy])
        # self.save_summary(summary_valid, train_step + conf.reload_step, mode='valid')
        if valid_acc > best_validation_accuracy:
            best_validation_accuracy = valid_acc
            improved_str = '(improved)'
            # self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''

        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}{3}'
              .format(epoch, valid_loss, valid_acc, improved_str))
        y_true = np.reshape(np.argmax(data_reader.y_valid, axis=-1), [-1])
        print(confusion_matrix(y_true, y_pred))
        print('-' * 60)

