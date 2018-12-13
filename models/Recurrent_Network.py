import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn.metrics import confusion_matrix
from DataLoaders.Sequential_ApoptosisLoader import DataLoader
import os
from tensorflow.python.ops.rnn import _transpose_batch_time
from config import args
from models.utils.metrics import *

from models.utils.metrics import compute_sequence_accuracy
from models.utils.plot_utils import visualize


class RecNet(object):
    def __init__(self, sess, conf, cnn_model):
        self.sess = sess
        self.conf = conf
        self.feature_extractor = cnn_model
        self.seqLen = tf.placeholder(tf.int32, [self.conf.batch_size])
        self.in_keep_prob = tf.placeholder(tf.float32, shape=())
        self.out_keep_prob = tf.placeholder(tf.float32, shape=())
        self.features = tf.reshape(self.feature_extractor.features, [conf.batch_size, conf.max_time, 512])
        self.summary_list = []
        self.build()
        self.configure_summary()

    def build(self):
        with tf.variable_scope('RecNet'):
            print('*' * 20)
            if self.conf.recurrent_model == 'RNN':
                print('RNN with {} layer(s) and {} hidden units generated'.
                      format(self.conf.num_layers, self.conf.num_hidden))
                cell = rnn.BasicRNNCell(self.conf.num_hidden)
                outputs, states = tf.nn.dynamic_rnn(cell, self.features, sequence_length=self.seqLen, dtype=tf.float32)
                weights = weight_variable(shape=[self.conf.num_hidden, self.conf.num_cls])
                biases = bias_variable(shape=[self.conf.num_cls])
                w_repeated = tf.tile(tf.expand_dims(weights, 0), [self.conf.batch_size, 1, 1])
                logits_temp = tf.matmul(outputs, w_repeated) + biases
            elif self.conf.recurrent_model == 'LSTM':
                print('LSTM with {} layer(s) and {} hidden units generated'.
                      format(self.conf.num_layers, self.conf.num_hidden))
                outputs = lstm(self.features, self.conf.num_layers, self.conf.num_hidden,
                               self.in_keep_prob, self.out_keep_prob)
                weights = weight_variable(shape=[self.conf.num_hidden[-1], self.conf.num_cls])
                biases = bias_variable(shape=[self.conf.num_cls])
                w_repeated = tf.tile(tf.expand_dims(weights, 0), [self.conf.batch_size, 1, 1])
                logits_temp = tf.matmul(outputs, w_repeated) + biases
            elif self.conf.recurrent_model == 'BiLSTM':
                print('Bidirectional LSTM with {} layer(s) and {} hidden units generated'.
                      format(self.conf.num_layers, self.conf.num_hidden))
                outputs = bidirectional_lstm(self.features, self.conf.num_layers,
                                             self.conf.num_hidden, self.in_keep_prob, self.out_keep_prob)
                weights = weight_variable(shape=[2 * self.conf.num_hidden[-1], self.conf.num_cls])
                biases = bias_variable(shape=[self.conf.num_cls])
                w_repeated = tf.tile(tf.expand_dims(weights, 0), [self.conf.batch_size, 1, 1])
                logits_temp = tf.matmul(outputs, w_repeated) + biases
            elif self.conf.recurrent_model == 'MANN':
                print('MANN with {} layer(s) and {} hidden units generated'.
                      format(self.conf.num_layers, self.conf.num_hidden))
                from mann_cell import MANNCell
                cell = MANNCell(self.conf.num_hidden[0], self.conf.memory_size, self.conf.memory_vector_dim,
                                head_num=args.read_head_num)
                state = cell.zero_state(args.batch_size, tf.float32)
                self.o = []
                for t in range(args.max_time):
                    output, state = cell(self.features[:, t, :], state)
                    with tf.variable_scope("o2o", reuse=(t > 0)):
                        o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], args.num_cls],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        tf.add_to_collection('reg_weights', o2o_w)
                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                        o2o_b = tf.get_variable('o2o_b', [args.num_cls],
                                                initializer=tf.contrib.layers.xavier_initializer())
                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                        output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)
                        # output = tf.nn.softmax(output, dim=1)
                    self.o.append(output)
                out = tf.stack(self.o, axis=1)
                logits_temp = tf.reshape(out, [self.conf.batch_size*self.conf.max_time, self.conf.num_cls])

            elif self.conf.recurrent_model == 'myrnn':
                print('myRNN with {} layer(s) and {} hidden units generated'.
                      format(self.conf.num_layers, self.conf.num_hidden))
                logits_temp = self.my_rnn(self.features, self.conf.num_layers, self.conf.num_hidden)
        print('*' * 20)
        self.logits = tf.reshape(logits_temp, [self.conf.batch_size * self.conf.max_time, self.conf.num_cls])
        self.y_pred_tensor = tf.argmax(self.logits, axis=-1, name='predictions')
        self.labels = tf.reshape(self.feature_extractor.y, [self.conf.batch_size * self.conf.max_time, self.conf.num_cls])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                                         logits=self.logits), name='loss')
        self.mean_loss, self.mean_loss_op = tf.metrics.mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam-op')
        train_vars = tf.trainable_variables()
        self.train_op = optimizer.minimize(loss, var_list=train_vars)

        correct_prediction = tf.equal(self.y_pred_tensor, tf.argmax(self.labels, -1), name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

        scope = 'CapsNet'
        # if self.conf.trainable:
        trained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[:16]  #############
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
        self.reload_cnn(self.conf.reload_step)
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
                             self.in_keep_prob: self.conf.in_keep_prob, self.out_keep_prob: self.conf.out_keep_prob,
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
                         self.in_keep_prob: 1, self.out_keep_prob: 1,
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

    def test(self, cnn_step_num, rnn_step_num):
        self.reload_cnn(cnn_step_num)
        self.reload_rnn(rnn_step_num)
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='test')
        self.num_test_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='test')
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        y_pred = np.zeros((self.data_reader.y_test.shape[0]) * self.conf.max_time)
        for step in range(self.num_test_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.feature_extractor.x: x_test, self.feature_extractor.y: y_test,
                         self.feature_extractor.is_training: False,
                         self.in_keep_prob: 1, self.out_keep_prob: 1,
                         self.seqLen: self.conf.max_time * np.ones(self.conf.batch_size)}
            yp, _, _ = self.sess.run([self.y_pred_tensor, self.mean_loss_op, self.mean_accuracy_op],
                                     feed_dict=feed_dict)
            y_pred[start * self.conf.max_time:end * self.conf.max_time] = yp
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.format(test_loss, test_acc))
        y_true = np.reshape(np.argmax(self.data_reader.y_test, axis=-1), [-1])
        print(confusion_matrix(y_true, y_pred))
        print('-' * 50)

        compute_sequence_accuracy(np.argmax(self.data_reader.y_test, axis=-1), np.reshape(y_pred, (-1, 72)))
        # import h5py
        # h5f = h5py.File('bilstm_alexnet.h5', 'w')
        # h5f.create_dataset('x', data=self.data_reader.x_test)
        # h5f.create_dataset('y_true', data=np.argmax(self.data_reader.y_test, axis=-1))
        # h5f.create_dataset('y_pred', data=np.reshape(y_pred, (-1, 72)))
        # h5f.close()

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.rnn_modeldir + self.conf.rnn_run_name, self.conf.rnn_model_name)
        self.rnn_saver.save(self.sess, checkpoint_path, global_step=step)

    def reload_rnn(self, step):
        checkpoint_path = os.path.join(self.conf.rnn_modeldir + self.conf.rnn_run_name, self.conf.rnn_model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the RNN model...')
        self.rnn_saver.restore(self.sess, model_path)
        print('----> RNN Model successfully restored')

    def reload_cnn(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the CNN model...')
        self.cnn_saver.restore(self.sess, model_path)
        print('----> CNN Model successfully restored')

    def my_rnn(self, input_data, num_layers, rnn_size):
        output = input_data
        ys = tf.reshape(self.feature_extractor.y, [self.conf.batch_size, self.conf.max_time, self.conf.num_cls])
        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer), reuse=tf.AUTO_REUSE):
                cell = tf.contrib.rnn.LSTMCell(rnn_size[layer])
                output, _ = sampling_rnn(cell,
                                         initial_state=cell.zero_state(self.conf.batch_size, dtype=tf.float32),
                                         input_=output, true_output=ys,
                                         seq_lengths=self.conf.max_time,
                                         is_train=self.feature_extractor.is_training)
        return output

    def grad_cam(self, cnn_step_num, rnn_step_num):
        self.reload_cnn(cnn_step_num)
        self.reload_rnn(rnn_step_num)
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='test')
        self.num_test_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='test')
        self.is_train = False
        self.prob = tf.nn.softmax(self.logits)
        self.sess.run(tf.local_variables_initializer())
        cost = (-1) * tf.reduce_sum(tf.multiply(self.feature_extractor.y, tf.log(self.prob)), axis=1)
        # gradient for partial linearization. We only care about target visualization class.
        y_c = tf.reduce_sum(tf.multiply(self.logits, self.feature_extractor.y), axis=1)   # vgg.fc8: outputs before softmax
        # Get last convolutional layer gradient for generating gradCAM visualization
        target_conv_layer = self.feature_extractor.net_grad   # vgg.pool5 of shape (batch_size, 7, 7, 512)
        target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, self.feature_extractor.x)[0]

        for step in range(self.num_test_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.feature_extractor.x: x_test, self.feature_extractor.y: y_test,
                         self.feature_extractor.is_training: False,
                         self.in_keep_prob: 1, self.out_keep_prob: 1,
                         self.seqLen: self.conf.max_time * np.ones(self.conf.batch_size)}
            prob, gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = self.sess.run(
                [self.prob, gb_grad, target_conv_layer, target_conv_layer_grad],
                feed_dict=feed_dict)
            visualize(x_test[::3], target_conv_layer_value[::3], target_conv_layer_grad_value[::3], gb_grad_value[::3],
                      prob[::3], y_test[::3], img_size=self.conf.height, fig_name='img_' + str(step))


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


def bidirectional_lstm(input_data, num_layers, rnn_size, in_keep_prob, out_keep_prob):
    output = input_data
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size[layer],
                                              initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=in_keep_prob
                                                    , output_keep_prob=out_keep_prob)
            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size[layer],
                                              initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
            cell_bw = rnn.DropoutWrapper(cell_bw, input_keep_prob=in_keep_prob, output_keep_prob=out_keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                              cell_bw,
                                                              output,
                                                              dtype=tf.float32)
            output = tf.concat(outputs, 2)
    return output


def lstm(input_data, num_layers, rnn_size, in_keep_prob, out_keep_prob):
    output = input_data
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer), reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.LSTMCell(rnn_size[layer])
            cell = rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob, output_keep_prob=out_keep_prob)
            output, states = tf.nn.dynamic_rnn(cell, output, dtype=tf.float32)
    return output


def sampling_rnn(cell, initial_state, input_, true_output, seq_lengths, is_train):
    # raw_rnn expects time major inputs as TensorArrays
    max_time = args.max_time  # this is the max time step per batch
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time, clear_after_read=False)
    inputs_ta = inputs_ta.unstack(_transpose_batch_time(input_))  # input_ is the input placeholder
    input_dim = input_.get_shape()[-1].value  # the dimensionality of the input to each time step
    output_dim = args.num_cls  # the dimensionality of the model's output at each time step

    def loop_fn(time, cell_output, cell_state, loop_state):
        """
        Loop function that allows to control input to the rnn cell and manipulate cell outputs.
        :param time: current time step
        :param cell_output: output from previous time step or None if time == 0
        :param cell_state: cell state from previous time step
        :param loop_state: custom loop state to share information between different iterations of this loop fn
        :return: tuple consisting of
          elements_finished: tensor of size [bach_size] which is True for sequences that have reached their end,
            needed because of variable sequence size
          next_input: input to next time step
          next_cell_state: cell state forwarded to next time step
          emit_output: The first return argument of raw_rnn. This is not necessarily the output of the RNN cell,
            but could e.g. be the output of a dense layer attached to the rnn layer.
          next_loop_state: loop state forwarded to the next time step
        """
        if cell_output is None:
            # time == 0, used for initialization before first call to cell
            next_cell_state = initial_state
            # the emit_output in this case tells TF how future emits look
            emit_output = tf.zeros([output_dim])
        else:
            # t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
            # here you can do whatever you want with cell_output before assigning it to emit_output.
            # In this case, we don't do anything
            next_cell_state = cell_state
            emit_output = tf.layers.dense(cell_output, 2,
                                          name='output_to_p',
                                          reuse=tf.AUTO_REUSE)

        # check which elements are finished
        elements_finished = (time >= seq_lengths)
        finished = tf.reduce_all(elements_finished)

        # assemble cell input for upcoming time step
        current_output = emit_output if cell_output is not None else None
        input_original = inputs_ta.read(time)  # tensor of shape (batch_size, input_dim)

        if current_output is None:
            # this is the initial step, i.e. there is no output from a previous time step, what we feed here
            # can highly depend on the data. In this case we just assign the actual input in the first time step.
            next_in = tf.concat((input_original, tf.zeros((args.batch_size, output_dim))), axis=-1)
        else:
            # time > 0, so just use previous output as next input
            # here you could do fancier things, whatever you want to do before passing the data into the rnn cell
            # if here you were to pass input_original than you would get the normal behaviour of dynamic_rnn
            when_train = tf.concat((input_original, true_output[:, time - 1, :]), axis=-1)
            when_test = tf.concat((input_original, current_output), axis=-1)
            next_in = tf.cond(is_train,
                              lambda: when_train,
                              lambda: when_test)

        next_input = tf.cond(finished,
                             lambda: tf.zeros([args.batch_size, input_dim + output_dim], dtype=tf.float32),
                             # copy through zeros
                             lambda: next_in)  # if not finished, feed the previous output as next input

        # set shape manually, otherwise it is not defined for the last dimensions
        next_input.set_shape([args.batch_size, input_dim + output_dim])

        # loop state not used in this example
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    outputs_ta, last_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = _transpose_batch_time(outputs_ta.stack())
    final_state = last_state

    return outputs, final_state
