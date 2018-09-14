from models.capsule_layers.Conv_Caps import ConvCapsuleLayer
from models.capsule_layers.FC_Caps import FCCapsuleLayer
from keras import layers
import tensorflow as tf
from tensorflow.contrib import rnn
from DataLoaders.Sequential_ApoptosisLoader import DataLoader
from config import args as conf
import numpy as np
from sklearn.metrics import confusion_matrix
import os

from models.utils.ops_cnn import *

num_hidden = 200
max_time = 72
num_cls = 2
batch_size = 4
input_dim = 512
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


def mask(yy, y_p, digit_c, is_train):  # used in capsule network
    with tf.variable_scope('Masking'):
        y_pred_ohe = tf.one_hot(y_p, depth=2)
        # [?, 10] (one-hot-encoded predicted labels)

        reconst_targets = tf.cond(is_train,  # condition
                                  lambda: yy,  # if True (Training)
                                  lambda: y_pred_ohe,  # if False (Test)
                                  name="reconstruction_targets")
        # [?, 10]
        out_masked = tf.multiply(digit_c, tf.expand_dims(reconst_targets, -1))
        # [?, 2, 16]
        return out_masked


def decoder(output_mask):
    with tf.variable_scope('Decoder'):
        decoder_input = tf.reshape(output_mask, [-1, conf.num_cls * conf.digit_caps_dim])
        # [?, 160]
        fc1 = tf.layers.dense(decoder_input, conf.h1, activation=tf.nn.relu, name="FC1", trainable=conf.trainable)
        # [?, 512]
        fc2 = tf.layers.dense(fc1, conf.h2, activation=tf.nn.relu, name="FC2", trainable=conf.trainable)
        # [?, 1024]
        decoder_output = tf.layers.dense(fc2, conf.width * conf.height * conf.channel,
                                         activation=tf.nn.sigmoid, name="FC3", trainable=conf.trainable)
        # [?, 784]
        return decoder_output


def reload_(sess_, step_, saver_):
    checkpoint_path = os.path.join(conf.modeldir + conf.run_name, conf.model_name)
    model_path = checkpoint_path + '-' + str(step_)
    if not os.path.exists(model_path + '.meta'):
        print('----> No such checkpoint found', model_path)
        return
    print('----> Restoring the CNN model...')
    saver_.restore(sess_, model_path)
    print('----> CNN Model successfully restored')


x = tf.placeholder(tf.float32, [batch_size * max_time, conf.height, conf.width, conf.channel])
y = tf.placeholder(tf.float32, [batch_size * max_time, num_cls])
seqLen = tf.placeholder(tf.int32, [batch_size])
is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

# with tf.variable_scope('CapsNet'):
#     net = lrn(relu(conv_layer(x, kernel_size=7, stride=2, num_filters=96, trainable=conf.trainable,
#                               add_reg=conf.L2_reg, layer_name='CONV1')))
#     net = max_pool(net, pool_size=3, stride=2, padding='SAME', name='MaxPool1')
#     net = lrn(relu(conv_layer(net, kernel_size=5, stride=2, num_filters=256, trainable=conf.trainable,
#                               add_reg=conf.L2_reg, layer_name='CONV2')))
#     net = max_pool(net, pool_size=3, stride=2, padding='SAME', name='MaxPool2')
#     net = relu(conv_layer(net, kernel_size=3, stride=1, num_filters=384, trainable=conf.trainable,
#                           add_reg=conf.L2_reg, layer_name='CONV3'))
#     net = relu(conv_layer(net, kernel_size=3, stride=1, num_filters=384, trainable=conf.trainable,
#                           add_reg=conf.L2_reg, layer_name='CONV4'))
#     net = relu(conv_layer(net, kernel_size=3, stride=1, num_filters=256, trainable=conf.trainable,
#                           add_reg=conf.L2_reg, layer_name='CONV5'))
#     net = max_pool(net, pool_size=3, stride=2, padding='SAME', name='MaxPool3')
#     layer_flat = flatten(net)
#     net = relu(fc_layer(layer_flat, num_units=512, add_reg=conf.L2_reg,
#                         trainable=conf.trainable, layer_name='FC1'))
#     net = dropout(net, conf.dropout_rate, training=is_training)
#     net = relu(fc_layer(net, num_units=512, add_reg=conf.L2_reg,
#                         trainable=conf.trainable, layer_name='FC2'))
#     net = dropout(net, conf.dropout_rate, training=is_training)
#     features = tf.reshape(net, [batch_size, max_time, 512])
#     cnn_logits = fc_layer(net, num_units=conf.num_cls, add_reg=conf.L2_reg,
#                           trainable=conf.trainable, layer_name='FC3')
with tf.variable_scope('CapsNet'):
    # Layer 1: A 2D conv layer
    conv1 = layers.Conv2D(filters=128, kernel_size=5, strides=2,
                          padding='valid', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x caps_dim(=filters)
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 2: Convolutional Capsule
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=8, caps_dim=16, strides=1, padding='same',
                                    routings=3, name='primarycaps')(conv1_reshaped)

    # Layer 3: Convolutional Capsule
    secondary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=8, caps_dim=16, strides=2, padding='same',
                                      trainable=conf.trainable, routings=3, name='secondarycaps')(primary_caps)
    _, H, W, D, dim = secondary_caps.get_shape()
    sec_cap_reshaped = layers.Reshape((H.value * W.value * D.value, dim.value))(secondary_caps)

    # Layer 4: Fully-connected Capsule
    digit_caps = FCCapsuleLayer(num_caps=conf.num_cls, caps_dim=conf.digit_caps_dim,
                                routings=3, name='digitcaps')(sec_cap_reshaped)
    # [?, 10, 16]

    epsilon = 1e-9
    v_length = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=2, keep_dims=True) + epsilon)
    # [?, 10, 1]
    act = tf.reshape(v_length, (conf.batch_size, conf.num_cls))

    y_prob_argmax = tf.to_int32(tf.argmax(v_length, axis=1))
    # [?, 1]
    y_pred = tf.squeeze(y_prob_argmax)
    # [?] (predicted labels)
    if conf.add_recon_loss:
        output_masked = decoder(mask(y, y_pred, digit_caps, is_training))

    if conf.before_mask:
        features = digit_caps
    else:
        features = output_masked
    features = tf.reshape(features, [conf.batch_size, conf.max_time, 32])
with tf.variable_scope('RecNet'):
    weights = weight_variable(shape=[num_hidden, num_cls])
    biases = bias_variable(shape=[num_cls])

cell = rnn.BasicLSTMCell(num_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, features, sequence_length=seqLen, dtype=tf.float32)
w_repeated = tf.tile(tf.expand_dims(weights, 0), [batch_size, 1, 1])
logits = tf.reshape(tf.matmul(outputs, w_repeated) + biases, [batch_size * max_time, num_cls])
y_pred_tensor = tf.argmax(logits, axis=-1, name='predictions')
labels = tf.reshape(y, [batch_size * max_time, num_cls])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                 logits=logits), name='loss')
# loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y,
#                                                                logits=logits,
#                                                                pos_weight=3), name='loss')
mean_loss, mean_loss_op = tf.metrics.mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adamop')
train_vars = tf.trainable_variables()
train_op = optimizer.minimize(loss, var_list=train_vars)

correct_prediction = tf.equal(y_pred_tensor, tf.argmax(labels, -1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
mean_accuracy, mean_accuracy_op = tf.metrics.mean(accuracy)

# if conf.trainable:
trained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CapsNet')[:13]
# else:
#     trained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CapsNet')
cnn_saver = tf.train.Saver(var_list=trained_vars, max_to_keep=1000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    reload_(sess, conf.reload_step, cnn_saver)
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
            feed_dict = {x: x_batch, y: y_batch, seqLen: max_time * np.ones(batch_size), is_training: True}
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
                         seqLen: max_time * np.ones(batch_size), is_training: False}
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
