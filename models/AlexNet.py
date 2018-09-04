from base_model import BaseModel
import tensorflow as tf
from models.utils.cnn_ops import conv_2d, flatten_layer, fc_layer, dropout, max_pool, lrn


class AlexNet(BaseModel):
    def __init__(self, sess, conf):
        super(AlexNet, self).__init__(sess, conf)
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            net = conv_2d(x, 7, 2, 96, 'CONV1', add_reg=self.conf.L2_reg)
            net = lrn(net)
            net = max_pool(net, 3, 2, 'MaxPool1')
            net = conv_2d(net, 5, 2, 256, 'CONV2', add_reg=self.conf.L2_reg)
            net = lrn(net)
            net = max_pool(net, 3, 2, 'MaxPool2')
            net = conv_2d(net, 3, 1, 384, 'CONV3', add_reg=self.conf.L2_reg)
            net = conv_2d(net, 3, 1, 384, 'CONV4', add_reg=self.conf.L2_reg)
            net = conv_2d(net, 3, 1, 256, 'CONV5', add_reg=self.conf.L2_reg)
            net = max_pool(net, 3, 2, 'MaxPool3')
            layer_flat = flatten_layer(net)
            net = fc_layer(layer_flat, 512, 'FC1', add_reg=self.conf.L2_reg)
            net = dropout(net, self.conf.keep_prob)
            net = fc_layer(net, 512, 'FC2', add_reg=self.conf.L2_reg)
            net = dropout(net, self.conf.keep_prob)
            self.logits = fc_layer(net, self.conf.num_cls, 'FC_out', use_relu=False, add_reg=self.conf.L2_reg)
            # [?, num_cls]
            self.probs = tf.nn.softmax(self.logits)
            # [?, num_cls]
            self.y_pred = tf.to_int32(tf.argmax(self.probs, 1))
            # [?] (predicted labels)
