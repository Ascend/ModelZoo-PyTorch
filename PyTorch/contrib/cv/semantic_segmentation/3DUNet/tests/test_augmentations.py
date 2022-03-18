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
# ============================================================================


from lib.medloaders.medical_loader_utils import generate_padded_subvolumes
import torch
import matplotlib.pyplot as plt
import lib.augment3D as augment

size = 32
from lib.medloaders.medical_image_process import load_medical_image

# t1 = torch.randn(size,size,size).numpy()
# t2  = torch.randn(size,size,size).numpy()
b = torch.randn(size, size, size).numpy()

t1 = load_medical_image('.././datasets/iseg_2017/iSeg-2017-Training/subject-1-T1.hdr').squeeze().numpy()
label = load_medical_image('.././datasets/iseg_2017/iSeg-2017-Training/subject-1-label.img').squeeze().numpy()
f, axarr = plt.subplots(4, 1)

axarr[0].imshow(t1[70, :, :])
axarr[1].imshow(label[70, :, :])

c = augment.RandomChoice(transforms=[augment.GaussianNoise(mean=0, std=0.1)])
[t1], label = c([t1], label)

axarr[2].imshow(t1[70, :, :])
axarr[3].imshow(label[70, :, :])

plt.show()
