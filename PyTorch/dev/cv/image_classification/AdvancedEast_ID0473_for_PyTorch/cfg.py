#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#

#npu modify begin
import os

#train_task_id = 'MBV3_512'
train_task_id = '3T256'
initial_epoch = 0
epoch_num = 1
lr = 1e-3
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 5
#load_weights = True
load_weights = False
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 10000
validation_split_ratio = 0.1
max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])  # 2400
assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'
if max_train_img_size == 256:
    batch_size = 8
elif max_train_img_size == 384:
    batch_size = 4
elif max_train_img_size == 512:
    batch_size = 2
else:
    batch_size = 1
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

pwd = os.getcwd()
data_dir = '/data/icpr/'
#data_dir = '/npu/traindata/ID0473_CarPeting_Pytorch_AdvancedEast/data/icpr/'

origin_image_dir_name = 'image_9000/'
origin_txt_dir_name = 'txt_9000/'
train_image_dir_name = 'images_%s/' % train_task_id
train_label_dir_name = 'labels_%s/' % train_task_id
show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
show_act_image_dir_name = 'show_act_images_%s/' % train_task_id
gen_origin_img = True
draw_gt_quad = True
draw_act_quad = True
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]
locked_layers = False

if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')

model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id
saved_model_weights_file_path = 'saved_model/east_model_weights_%s.h5'\
                                 % train_task_id
#saved_model_weights_file_path ='model/mbv3_weights_3T640.004-0.156.h5'
pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
#predict_write2txt = False
predict_write2txt = True
#npu modify end
