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

            self.logits = fc_layer(net, self.conf.num_cls, 'FC_out', use_relu=False, add_reg=self.conf.L2_reg)
            # [?, num_cls]
            self.probs = tf.nn.softmax(self.logits)
            # [?, num_cls]
            self.y_pred = tf.to_int32(tf.argmax(self.probs, 1))
            # [?] (predicted labels)
