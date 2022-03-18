# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import time

if len(sys.argv) < 2:
    print("Usage: python visual_tf_model.py <model.pb>")
    sys.exit(0)

model_file_name = sys.argv[1]
with tf.Session() as sess:
    with gfile.FastGFile(model_file_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='log'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

while True:
    time.sleep(1000)