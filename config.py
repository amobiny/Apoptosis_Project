import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', 'train_sequence', 'train, train_sequence or test')
flags.DEFINE_integer('step_num', 0, 'model number to load')
flags.DEFINE_string('model', 'alexnet', 'alexnet, resnet, densenet, original_capsule, '
                                                 'fast_capsule, matrix_capsule or vector_capsule')
flags.DEFINE_string('loss_type', 'cross_entropy', 'cross_entropy, spread or margin')
flags.DEFINE_boolean('add_recon_loss', False, 'To add reconstruction loss')
flags.DEFINE_boolean('L2_reg', False, 'Adds L2-regularization to all the network weights')
flags.DEFINE_float('lmbda', 5e-04, 'L2-regularization coefficient')

# Training logs
flags.DEFINE_integer('max_step', 100000, '# of step for training (only for mnist)')
flags.DEFINE_integer('max_epoch', 1000, '# of step for training (only for nodule data)')
flags.DEFINE_boolean('epoch_based', True, 'Running the training in epochs')
flags.DEFINE_integer('SAVE_FREQ', 1000, 'Number of steps to save model')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 500, 'Number of step to evaluate the network on Validation data')

# For margin loss
flags.DEFINE_float('m_plus', 0.9, 'm+ parameter')
flags.DEFINE_float('m_minus', 0.1, 'm- parameter')
flags.DEFINE_float('lambda_val', 0.5, 'Down-weighting parameter for the absent class')
# For reconstruction loss
flags.DEFINE_float('alpha', 0.0005, 'Regularization coefficient to scale down the reconstruction loss')
# For training
flags.DEFINE_integer('batch_size', 32, 'training batch size')
flags.DEFINE_float('init_lr', 1e-4, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# data
flags.DEFINE_string('data', 'apoptosis', 'mnist or nodule or cifar10 or apoptosis')
flags.DEFINE_integer('num_cls', 2, 'Number of output classes')
flags.DEFINE_integer('N', 72000, 'Total number of training samples')
flags.DEFINE_float('percent', 1, 'Percentage of training data to use')
flags.DEFINE_integer('dim', 2, '2D or 3D for nodule data')
flags.DEFINE_boolean('one_hot', False, 'one-hot-encodes the labels')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_boolean('flip', False, 'Flips the data left to right and side to side')
flags.DEFINE_integer('max_angle', 180, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('height', 28, 'Network input height size')
flags.DEFINE_integer('width', 28, 'Network input width size')
flags.DEFINE_integer('depth', 32, 'Network input depth size (in the case of 3D input images)')
flags.DEFINE_integer('channel', 1, 'Network input channel size')

# Directories
flags.DEFINE_string('run_name', 'alex_1', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Saved models directory')
flags.DEFINE_integer('reload_step', 33749, 'Reload step to continue training')
flags.DEFINE_string('model_name', 'model', 'Model file name')

# CNN architecture
flags.DEFINE_float('dropout_rate', 0.2, 'The dropout rate, between 0 and 1. '
                                        'E.g. "rate=0.1" would drop out 10% of input units')
flags.DEFINE_integer('growth_rate', 24, 'Growth rate of DenseNet')
flags.DEFINE_integer('num_levels', 3, '# of levels (dense block + Transition Layer) in DenseNet')
flags.DEFINE_list('num_BBs', [6, 8, 10], '# of bottleneck-blocks at each level')
flags.DEFINE_float('theta', 1, 'Compression factor in DenseNet')

# CapsNet architecture
flags.DEFINE_integer('prim_caps_dim', 8, 'Dimension of the PrimaryCaps in the Original_CapsNet')
flags.DEFINE_integer('digit_caps_dim', 16, 'Dimension of the DigitCaps in the Original_CapsNet')
flags.DEFINE_integer('h1', 512, 'Number of hidden units of the first FC layer of the reconstruction network')
flags.DEFINE_integer('h2', 1024, 'Number of hidden units of the second FC layer of the reconstruction network')

# Matrix Capsule architecture
flags.DEFINE_boolean('use_bias', True, 'Adds bias to init capsules')
flags.DEFINE_boolean('use_BN', True, 'Adds BN before conv1 layer')
flags.DEFINE_boolean('add_coords', True, 'Adds capsule coordinations')
flags.DEFINE_boolean('grad_clip', False, 'Adds gradient clipping to get rid of exploding gradient')
flags.DEFINE_integer('iter', 1, 'Number of EM-routing iterations')
flags.DEFINE_integer('A', 32, 'A in Figure 1 of the paper')
flags.DEFINE_integer('B', 16, 'B in Figure 1 of the paper')
flags.DEFINE_integer('C', 8, 'C in Figure 1 of the paper')
flags.DEFINE_integer('D', 8, 'D in Figure 1 of the paper')

# RNN architecture
flags.DEFINE_boolean('trainable', False, 'Whether to train the CNN or not')
flags.DEFINE_string('recurrent_model', 'LSTM', 'RNN, LSTM, BiRNN, and MANN')
flags.DEFINE_integer('num_hidden', 200, 'Number of hidden units for the Recurrent structure')
flags.DEFINE_integer('max_time', 72, 'Maximum length of sequences')

flags.DEFINE_string('rnn_run_name', 'test_0', 'Run name')
flags.DEFINE_string('rnn_logdir', './Results_recurrent/log_dir/', 'Logs directory')
flags.DEFINE_string('rnn_modeldir', './Results_recurrent/model_dir/', 'Saved models directory')
flags.DEFINE_integer('rnn_reload_step', 0, 'Reload step to continue training')
flags.DEFINE_string('rnn_model_name', 'model', 'Model file name')
args = tf.app.flags.FLAGS
