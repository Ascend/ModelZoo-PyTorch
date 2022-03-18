
from npu_bridge.npu_init import *
#from npu_bridge import *
import tensorflow as tf
#from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt

class TCNNConfig(object):
    'CNN配置参数'
    embedding_dim = 64
    seq_length = 600
    num_classes = 10
    num_filters = 256
    kernel_size = 5
    vocab_size = 5000
    hidden_dim = 128
    dropout_keep_prob = 0.5
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10
    print_per_batch = 100
    save_per_batch = 10

class TextCNN(object):
    '文本分类，CNN模型'

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        'CNN模型'
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        with tf.name_scope('cnn'):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        with tf.name_scope('score'):
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)).minimize(self.loss)
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
