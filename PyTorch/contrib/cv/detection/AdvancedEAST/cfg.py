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

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--apex', action='store_true', help='Use apex for mixed precision training')
parser.add_argument('--device', default='npu', help='cpu npu')
parser.add_argument('--size', default=736, type=int)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--img_path', default='demo/004.jpg')
parser.add_argument('--threshold', default=0.9, type=float)
parser.add_argument('--pth_path')
parser.add_argument('--epoch_num', default=60, type=int)
parser.add_argument('--val_interval', default=3, type=int)

args = parser.parse_args()
amp = args.apex
device = args.device
local_rank = args.local_rank
img_path = args.img_path
predict_threshold = args.threshold
pth_path = args.pth_path
distributed = False if local_rank == -1 else True
is_master_node = True if local_rank < 1 else False
world_size = int(os.environ['WORLD_SIZE']) if distributed else 1

train_task_id = '3T' + str(args.size)
if is_master_node:
    print(train_task_id)
initial_epoch = 0
epoch_num = args.epoch_num
lr = 5e-4
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
workers = 16
patience = 5
val_interval = args.val_interval
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
batch_size = batch_size * 4
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

data_dir = 'icpr/'
origin_image_dir_name = 'image_10000/'
origin_txt_dir_name = 'txt_10000/'
train_image_dir_name = 'images_%s/' % train_task_id
train_label_dir_name = 'labels_%s/' % train_task_id
show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
show_act_image_dir_name = 'show_act_images_%s/' % train_task_id
lmdb_trainset_dir_name = data_dir + 'Lmdb_trainset_%s/' % train_task_id
lmdb_valset_dir_name = data_dir + 'Lmdb_valset_%s/' % train_task_id
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
locked_layers = False  # 是否冻结前两层参数

if not os.path.exists('saved_model'):
    os.makedirs('saved_model', exist_ok=True)

saved_model = ''
model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id
saved_model_weights_file_path = 'saved_model/adEAST_iter_%s.pth'\
                                % str(epoch_num + 1)

pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
iou_threshold = 0.5
predict_cut_text_line = False
predict_write2txt = True
model_summary = False
quiet = True
