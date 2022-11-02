# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
exp_name = 'qvi'
record_dir = 'records/{}'.format(exp_name)
checkpoint_dir = 'checkpoints/{}'.format(exp_name)
checkpoint = 'checkpoints/gpu_74.ckpt'
trainset = 'QVI960'
trainset_root = '/QVI-960'
train_size = (640, 360)
train_crop_size = (355, 355)

validationset = 'Adobe240all'
validationset_root = '/Adobe240_validation'
validation_size = (640, 360)
validation_crop_size = (640, 360)

train_batch_size = 14

train_continue = True
epochs = 250
progress_iter = 200
checkpoint_epoch = 1


mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]

model = 'QVI'
pwc_path = 'replace with obs path'

init_learning_rate = 1e-4
milestones = [100, 150]

