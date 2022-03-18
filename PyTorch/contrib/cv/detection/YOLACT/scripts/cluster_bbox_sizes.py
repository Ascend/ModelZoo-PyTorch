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
"""
This script compiles all the bounding boxes in the training data and
clusters them for each convout resolution on which they're used.

Run this script from the Yolact root directory.
"""

import os.path as osp
import json, pickle
import sys

import numpy as np
import sklearn.cluster as cluster

dump_file = 'weights/bboxes.pkl'
max_size = 550

num_scale_clusters = 5
num_aspect_ratio_clusters = 3

def to_relative(bboxes):
	return bboxes[:, 2:4] / bboxes[:, :2]

def process(bboxes):
	return to_relative(bboxes) * max_size

if __name__ == '__main__':
		
	with open(dump_file, 'rb') as f:
		bboxes = pickle.load(f)

	bboxes = np.array(bboxes)
	bboxes = process(bboxes)
	bboxes = bboxes[(bboxes[:, 0] > 1) * (bboxes[:, 1] > 1)]

	scale  = np.sqrt(bboxes[:, 0] * bboxes[:, 1]).reshape(-1, 1)

	clusterer = cluster.KMeans(num_scale_clusters, random_state=99, n_jobs=4)
	assignments = clusterer.fit_predict(scale)
	counts = np.bincount(assignments)

	cluster_centers = clusterer.cluster_centers_

	center_indices = list(range(num_scale_clusters))
	center_indices.sort(key=lambda x: cluster_centers[x, 0])

	for idx in center_indices:
		center = cluster_centers[idx, 0]
		boxes_for_center = bboxes[assignments == idx]
		aspect_ratios = (boxes_for_center[:,0] / boxes_for_center[:,1]).reshape(-1, 1)

		c = cluster.KMeans(num_aspect_ratio_clusters, random_state=idx, n_jobs=4)
		ca = c.fit_predict(aspect_ratios)
		cc = np.bincount(ca)

		c = list(c.cluster_centers_.reshape(-1))
		cidx = list(range(num_aspect_ratio_clusters))
		cidx.sort(key=lambda x: -cc[x])

		# import code
		# code.interact(local=locals())

		print('%.3f (%d) aspect ratios:' % (center, counts[idx]))
		for idx in cidx:
			print('\t%.2f (%d)' % (c[idx], cc[idx]))
		print()
		# exit()


