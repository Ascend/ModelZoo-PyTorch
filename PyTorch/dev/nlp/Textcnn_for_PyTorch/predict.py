
from __future__ import print_function
from npu_bridge.npu_init import *
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab
try:
    bool(type(unicode))
except NameError:
    unicode = str
base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')

class CnnModel():

    def __init__(self):
        self.config = TCNNConfig()
        (self.categories, self.cat_to_id) = read_category()
        (self.words, self.word_to_id) = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)

    def predict(self, message):
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if (x in self.word_to_id)]
        feed_dict = {self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length), self.model.keep_prob: 1.0}
        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]
if (__name__ == '__main__'):
    cnn_model = CnnModel()
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机', '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    for i in test_demo:
        print(cnn_model.predict(i))
