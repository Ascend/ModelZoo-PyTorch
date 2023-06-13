# coding:utf-8
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


import torch
from sklearn.decomposition import PCA
import numpy as np, os
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, Range1d, LabelSet
import h5py

#cd "/Users/fineday/Github/t2vec/"

folder = "experiment"
v = torch.load(os.path.join(folder, "trj.pt"))
label = np.loadtxt(os.path.join(folder, "trj.label")).astype(np.int32)
labelmap = {x: i for (i, x) in enumerate(np.unique(label))}
#colormap = ["#%02x%02x%02x"%(r*10, r*20, r*15) for r in range(50, 50+len(labelmap))]
#colors = [colormap[labelmap[x]] for x in label]
#colormap = {1: 'red', 2:'green', 3:'blue', 4:"purple"}
#colors = [colormap[x] for x in label]
v1, v2 = v[0], v[1]
v1, v2 = v1.squeeze(0), v2.squeeze(0)
v12 = torch.cat([v1, v2], dim=1)
v1, v2, v12 = v1.numpy(), v2.numpy(), v12.numpy()

## ----------------------------------------------

folder= "experiment"
assert os.path.isfile(os.path.join(folder, "trj.h5")), "trj.h5 does not exist"
with h5py.File(os.path.join(folder, "trj.h5"), "r") as f:
    v1, v2 = f["layer1"][...], f["layer2"][...]
    v12 = np.concatenate([v1, v2], axis=1)
label = np.loadtxt(os.path.join(folder, "trj.label")).astype(np.int32)
labelmap = {x: i for (i, x) in enumerate(np.unique(label))}



pca = PCA(n_components=2)
emb = pca.fit_transform(v1)

source = ColumnDataSource(data=dict(x=emb[:,0], y=emb[:,1],
                          tripname=[str(labelmap[x]) for x in label]))


labels = LabelSet(x='x', y='y', text='tripname', level='glyph',
    x_offset=5, y_offset=5, render_mode='canvas', source=source)

p = figure(x_range=(-10, 15))
p.scatter(x='x', y='y',  size=8, source=source)
p.add_layout(labels)
show(p)
