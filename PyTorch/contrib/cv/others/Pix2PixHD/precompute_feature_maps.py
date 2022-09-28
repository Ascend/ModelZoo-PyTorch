# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import os
import util.util as util
from torch.autograd import Variable
import torch.nn as nn

opt = TrainOptions().parse()
opt.nThreads = 1
opt.batchSize = 1 
opt.serial_batches = True 
opt.no_flip = True
opt.instance_feat = True

name = 'features'
save_path = os.path.join(opt.checkpoints_dir, opt.name)

############ Initialize #########
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = create_model(opt)
util.mkdirs(os.path.join(opt.dataroot, opt.phase + '_feat'))

######## Save precomputed feature maps for 1024p training #######
for i, data in enumerate(dataset):
	print('%d / %d images' % (i+1, dataset_size)) 
	feat_map = model.module.netE.forward(Variable(data['image'].cuda(), volatile=True), data['inst'].cuda())
	feat_map = nn.Upsample(scale_factor=2, mode='nearest')(feat_map)
	image_numpy = util.tensor2im(feat_map.data[0])
	save_path = data['path'][0].replace('/train_label/', '/train_feat/')
	util.save_image(image_numpy, save_path)