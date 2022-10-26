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
import numpy as np
import os

opt = TrainOptions().parse()
opt.nThreads = 1
opt.batchSize = 1 
opt.serial_batches = True 
opt.no_flip = True
opt.instance_feat = True
opt.continue_train = True

name = 'features'
save_path = os.path.join(opt.checkpoints_dir, opt.name)

############ Initialize #########
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = create_model(opt)

########### Encode features ###########
reencode = True
if reencode:
	features = {}
	for label in range(opt.label_nc):
		features[label] = np.zeros((0, opt.feat_num+1))
	for i, data in enumerate(dataset):    
	    feat = model.module.encode_features(data['image'], data['inst'])
	    for label in range(opt.label_nc):
	    	features[label] = np.append(features[label], feat[label], axis=0) 
	        
	    print('%d / %d images' % (i+1, dataset_size))    
	save_name = os.path.join(save_path, name + '.npy')
	np.save(save_name, features)

############## Clustering ###########
n_clusters = opt.n_clusters
load_name = os.path.join(save_path, name + '.npy')
features = np.load(load_name).item()
from sklearn.cluster import KMeans
centers = {}
for label in range(opt.label_nc):
	feat = features[label]
	feat = feat[feat[:,-1] > 0.5, :-1]		
	if feat.shape[0]:
		n_clusters = min(feat.shape[0], opt.n_clusters)
		kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feat)
		centers[label] = kmeans.cluster_centers_
save_name = os.path.join(save_path, name + '_clustered_%03d.npy' % opt.n_clusters)
np.save(save_name, centers)
print('saving to %s' % save_name)