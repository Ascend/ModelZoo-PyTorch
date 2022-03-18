# Copyright 2021 Huawei Technologies Co., Ltd
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

import torch
from apex import amp
# 指定具体的设备
torch.npu.set_device(0)

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'config/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = 'npu_8p_full/best_top1_acc_epoch_40.pth'

# 指定设备
device = 'npu:0'
device = torch.device(device)

# 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device)
model = amp.initialize(model.npu(), opt_level='O1', loss_scale=128.0)

# 测试单个视频并显示其结果
video = '/home/i3d_wtc/mmaction2/demo/demo.mp4'  # 这里指定要测试的单个视频
labels = '/home/i3d_wtc/mmaction2/tools/data/kinetics/label_map_k400.txt'  # kinetics400数据集的标签
results = inference_recognizer(model, video, labels)  # 在线推理的结果

# 显示结果
labels = open('/home/i3d_wtc/mmaction2/tools/data/kinetics/label_map_k400.txt').readlines()
labels = [x.strip() for x in labels]
results = [(k[0], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
