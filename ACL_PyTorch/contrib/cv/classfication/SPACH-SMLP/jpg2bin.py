# Copyright 2020 Huawei Technologies Co., Ltd
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

from easydict import EasyDict
from datasets import build_dataset
args = EasyDict(aa='rand-m9-mstd0.5-inc1', auto_resume=True, batch_size=128, clip_grad=None, color_jitter=0.4, cooldown_epochs=10, cutmix=1.0, cutmix_minmax=None, data_path='/opt/npu/imagenet/', data_set='IMNET', decay_epochs=30, decay_rate=0.1, device='cuda', dist_eval=True, dist_url='env://', distillation_alpha=0.5, distillation_tau=1.0, distillation_type='none', distributed=False, drop=0.0, drop_path=0.1, epochs=300, eval=False, finetune='', inat_category='name', input_size=224, lr=0.0005, lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, min_lr=1e-05, mixup=0.8, mixup_mode='batch', mixup_prob=1.0, mixup_switch_prob=0.5, model='smlpnet_tiny', model_ema=True, model_ema_decay=0.99996, model_ema_force_cpu=False, momentum=0.9, nb_classes=1000, num_workers=10, opt='adamw', opt_betas=None, opt_eps=1e-08, output_dir='./output2/', patience_epochs=10, pin_mem=True, recount=1, remode='pixel', repeated_aug=True, reprob=0.25, resplit=False, resume='', sched='cosine', seed=0, shared_spatial_func=False, smoothing=0.1, start_epoch=0, stem_type='conv1', teacher_model='regnety_160', teacher_path='', throughput=False, train_interpolation='bicubic', warmup_epochs=20, warmup_lr=1e-06, weight_decay=0.05, world_size=1)
dataset_val, _ = build_dataset(is_train=False, args=args)
n = len(dataset_val)
print(_, n)

import numpy as np
import os
import pandas as pd
from tqdm import tqdm

save_path = "./imagenet/val-bin/"
file_names = []
ids = []
Hs = []
Ws = []
for i in tqdm(range(n)):
	# print(dataset_val[i])
	img, label = dataset_val[i]
	img = img.numpy()

	# file_dir = os.path.join(save_path, f"{label:05d}")
	file_dir = save_path
	file_path = os.path.join(file_dir, f"{i}.bin")
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)

	img.tofile(file_path)

	file_names.append(file_path)
	ids.append(i)
	Hs.append(224)
	Ws.append(224)
	# if i > 100:
	# 	break

df = pd.DataFrame({"ids":ids, "file_names":file_names,
				"Hs":Hs, "Ws":Ws})

df.to_csv("./imagenet/imagenet-val.info", sep=" ", header=None, columns=None, index=None)