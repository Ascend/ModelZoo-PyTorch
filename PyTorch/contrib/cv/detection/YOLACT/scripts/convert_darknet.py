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
from backbone import DarkNetBackbone
import h5py
import torch

f = h5py.File('darknet53.h5', 'r')
m = f['model_weights']

yolo_keys = list(m.keys())
yolo_keys = [x for x in yolo_keys if len(m[x].keys()) > 0]
yolo_keys.sort()

sd = DarkNetBackbone().state_dict()

sd_keys = list(sd.keys())
sd_keys.sort()

# Note this won't work if there are 10 elements in some list but whatever that doesn't happen
layer_keys = list(set(['.'.join(x.split('.')[:-2]) for x in sd_keys]))
layer_keys.sort()

# print([x for x in sd_keys if x.startswith(layer_keys[0])])

mapping = {
	'.0.weight'      : ('conv2d_%d', 'kernel:0'),
	'.1.bias'        : ('batch_normalization_%d', 'beta:0'),
	'.1.weight'      : ('batch_normalization_%d', 'gamma:0'),
	'.1.running_var' : ('batch_normalization_%d', 'moving_variance:0'),
	'.1.running_mean': ('batch_normalization_%d', 'moving_mean:0'),
	'.1.num_batches_tracked': None,
}

for i, layer_key in zip(range(1, len(layer_keys) + 1), layer_keys):
	# This is pretty inefficient but I don't care
	for weight_key in [x for x in sd_keys if x.startswith(layer_key)]:
		diff = weight_key[len(layer_key):]
		
		if mapping[diff] is not None:
			yolo_key = mapping[diff][0] % i
			sub_key  = mapping[diff][1]

			yolo_weight = torch.Tensor(m[yolo_key][yolo_key][sub_key].value)
			if (len(yolo_weight.size()) == 4):
				yolo_weight = yolo_weight.permute(3, 2, 0, 1).contiguous()
			
			sd[weight_key] = yolo_weight

torch.save(sd, 'weights/darknet53.pth')

