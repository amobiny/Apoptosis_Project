import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn.metrics import confusion_matrix
from DataLoaders.Sequential_ApoptosisLoader import DataLoader
import os


class RecNet(object):
    def __init__(self, sess, conf, cnn_model):
        self.sess = sess
        self.conf = conf
        self.feature_extractor = cnn_model
        self.seqLen = tf.placeholder(tf.int32, [self.conf.batch_size])
        self.features = tf.reshape(self.feature_extractor.features, [conf.batch_size, conf.max_time, 32])
        self.summary_list = []
        self.build()
        self.configure_summary()

    def build(self):
        with tf.variable_scope('RecNet'):
            print('*'*20)
            if self.conf.recurrent_model == 'RNN':
                print('RNN with {} layers and {} hidden units generated'.
                      format(self.conf.num_layers, self.conf.num_hidden))
                cell = rnn.BasicRNNCell(self.conf.num_hidden)
                outputs, states = tf.nn.dynamic_rnn(cell, self.features, sequence_length=self.seqLen, dtype=tf.float32)
                weights = weight_variable(shape=[self.conf.num_hidden, self.conf.num_cls])
                biases = bias_variable(shape=[self.conf.num_cls])
            elif self.conf.recurrent_model == 'LSTM':
                print('LSTM with {} layers and {} hidden units generated'.
                      format(self.conf.num_layers, self.conf.num_hidden))
                outputs = lstm(self.features, self.conf.num_layers, self.conf.num_hidden, self.conf.dropout_rate)
                # cell = rnn.BasicLSTMCell(self.conf.num_hidden)
                # cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.conf.num_layers)])
                # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.conf.num_layers)
                # outputs, states = tf.nn.dynamic_rnn(cell, self.features, sequence_length=self.seqLen, dtype=tf.float32)
                weights = weight_variable(shape=[self.conf.num_hidden, self.conf.num_cls])
                biases = bias_variable(shape=[self.conf.num_cls])
            elif self.conf.recurrent_model == 'BiLSTM':
                print('Bidirectional LSTM with {} layers and {} hidden units generated'.
                      format(self.conf.num_layers, self.conf.num_hidden))
                outputs = bidirectional_lstm(self.features, self.conf.num_layers,
                                             self.conf.num_hidden, self.conf.dropout_rate)
                weights = weight_variable(shape=[2*self.conf.num_hidden, self.conf.num_cls])
                biases = bias_variable(shape=[self.conf.num_cls])
        print('*' * 20)
        w_repeated = tf.tile(tf.expand_dims(weights, 0), [self.conf.batch_size, 1, 1])
        logits = tf.reshape(tf.matmul(outputs, w_repeated) + biases,
                            [self.conf.batch_size * self.conf.max_time, self.conf.num_cls])
        self.y_pred_tensor = tf.argmax(logits, axis=-1, name='predictions')
        labels = tf.reshape(self.feature_extractor.y, [self.conf.batch_size * self.conf.max_time, self.conf.num_cls])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                         logits=logits), name='loss')
        self.mean_loss, self.mean_loss_op = tf.metrics.mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam-op')
        train_vars = tf.trainable_variables()
        self.train_op = optimizer.minimize(loss, var_list=train_vars)

        correct_prediction = tf.equal(self.y_pred_tensor, tf.argmax(labels, -1), name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

        scope = 'CapsNet'
        # if self.conf.trainable:
        trained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[:11]  #############
        # AlexNet: 16, ResNet: 426, CapsNet: 11,
        self.cnn_saver = tf.train.Saver(var_list=trained_vars, max_to_keep=1000)
        self.rnn_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1000)

    def configure_summary(self):
        self.train_writer = tf.summary.FileWriter(self.conf.rnn_logdir + self.conf.rnn_run_name + '/train/',
                                                  self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.rnn_logdir + self.conf.rnn_run_name + '/valid/')
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        tf.summary.scalar('Accuracy/average_accuracy', self.mean_accuracy)] + self.summary_list
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        # print('----> Summarizing at step {}'.format(step))
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        elif mode == 'valid':
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        self.reload_cnn()
        self.best_validation_accuracy = 0
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='train')
        self.data_reader.get_data(mode='valid')
        self.num_train_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='train')
        self.num_val_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='valid')
        for epoch in range(self.conf.max_epoch):
            self.data_reader.randomize()
            for train_step in range(self.num_train_batch):
                glob_step = epoch * self.num_train_batch + train_step
                start = train_step * self.conf.batch_size
                end = (train_step + 1) * self.conf.batch_size
                x_batch, y_batch = self.data_reader.next_batch(start, end, mode='train')
                feed_dict = {self.feature_extractor.x: x_batch, self.feature_extractor.y: y_batch,
                             self.feature_extractor.is_training: True,
                             self.seqLen: self.conf.max_time * np.ones(self.conf.batch_size)}
                if train_step % self.conf.SUMMARY_FREQ == 0:
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    self.save_summary(summary, glob_step + self.conf.reload_step, mode='train')
                    print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                else:
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            self.evaluate(glob_step)

    def evaluate(self, train_step):
        self.sess.run(tf.local_variables_initializer())
        y_pred = np.zeros((self.data_reader.y_valid.shape[0]) * self.conf.max_time)
        for step in range(self.num_val_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.feature_extractor.x: x_val, self.feature_extractor.y: y_val,
                         self.feature_extractor.is_training: False,
                         self.seqLen: self.conf.max_time * np.ones(self.conf.batch_size)}
            yp, _, _ = self.sess.run([self.y_pred_tensor, self.mean_loss_op, self.mean_accuracy_op],
                                     feed_dict=feed_dict)
            y_pred[start * self.conf.max_time:end * self.conf.max_time] = yp
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, train_step + self.conf.reload_step, mode='valid')
        if valid_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = valid_acc
            improved_str = '(improved)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''

        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}{3}'
              .format(train_step, valid_loss, valid_acc, improved_str))
        y_true = np.reshape(np.argmax(self.data_reader.y_valid, axis=-1), [-1])
        print(confusion_matrix(y_true, y_pred))
        print('-' * 60)

    def test(self, step_num):
        self.reload(step_num)
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='test')
        self.num_test_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='test')
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        y_pred = np.zeros((self.data_reader.y_test.shape[0]))
        for step in range(self.num_test_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.x: x_test, self.y: y_test, self.feature_extractor.is_training: False}
            yp, _, _ = self.sess.run([self.y_pred, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            y_pred[start:end] = yp
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.format(test_loss, test_acc))
        print(confusion_matrix(np.argmax(self.data_reader.y_test, axis=1), y_pred))
        print('-' * 50)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.rnn_modeldir + self.conf.rnn_run_name, self.conf.rnn_model_name)
        self.rnn_saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.rnn_modeldir + self.conf.rnn_run_name, self.conf.rnn_model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')

    def reload_cnn(self):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(self.conf.reload_step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the CNN model...')
        self.cnn_saver.restore(self.sess, model_path)
        print('----> CNN Model successfully restored')


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


def bidirectional_lstm(input_data, num_layers, rnn_size, keep_prob):
    output = input_data
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            # cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)
            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            # cell_bw = rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                              cell_bw,
                                                              output,
                                                              dtype=tf.float32)
            output = tf.concat(outputs, 2)
    return output


def lstm(input_data, num_layers, rnn_size, keep_prob):
    output = input_data
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer), reuse=tf.AUTO_REUSE):
            # cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            cell = tf.contrib.rnn.LSTMCell(rnn_size)
            # cell = rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
            output, _ = tf.nn.dynamic_rnn(cell, output, dtype=tf.float32)
    return output
