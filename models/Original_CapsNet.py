from models.base_model import BaseModel
from models.capsule_layers.FC_Caps import FCCapsuleLayer
# from keras import layers
import tensorflow as tf
from models.capsule_layers.ops import squash


class Orig_CapsNet(BaseModel):
    def __init__(self, sess, conf):
        super(Orig_CapsNet, self).__init__(sess, conf)
        self.build_network(self.x)
        if self.conf.mode != 'train_sequence' and self.conf.mode != 'test_sequence':
            self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            # Layer 1: A 2D conv layer
            conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=1, trainable=self.conf.trainable,
                                           padding='valid', activation='relu', name='conv1')(x)

            # conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, trainable=self.conf.trainable,
            #                                padding='valid', activation='relu', name='conv2')(conv1)

            # Layer 2: Primary Capsule Layer; simply a 2D conv + reshaping
            primary_caps = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=2, trainable=self.conf.trainable,
                                         padding='valid', activation='relu', name='primary_caps')(conv1)
            _, H, W, dim = primary_caps.get_shape()
            num_caps = int(H.value * W.value * dim.value / self.conf.prim_caps_dim)
            primary_caps_reshaped = tf.keras.layers.Reshape((num_caps, self.conf.prim_caps_dim))(primary_caps)
            caps1_output = squash(primary_caps_reshaped)

            # Layer 3: Digit Capsule Layer; Here is where the routing takes place
            self.digit_caps = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=self.conf.digit_caps_dim,
                                             routings=3, name='digit_caps', trainable=self.conf.trainable)(caps1_output)
            # [?, 2, 16]

            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keep_dims=True) + epsilon)
            # [?, 2, 1]
            self.act = tf.reshape(self.v_length, (-1, self.conf.num_cls))
            self.prob = tf.nn.softmax(self.act)
            y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
            # [?, 1]
            self.y_pred = tf.squeeze(y_prob_argmax)
            # [?] (predicted labels)

            if self.conf.add_recon_loss:
                self.mask()
                self.decoder()

            if self.conf.before_mask:
                self.features = self.digit_caps
            else:
                self.features = self.output_masked

            self.net_grad = primary_caps
            self.logits = self.act