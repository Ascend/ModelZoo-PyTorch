#
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
#

import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib import rc
from .draw_utils import COLOR, MARKER_STYLE

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def draw_eao(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    angles = np.linspace(0, 2*np.pi, 8, endpoint=True)

    attr2value = []
    for i, (tracker_name, ret) in enumerate(result.items()):
        value = list(ret.values())
        attr2value.append(value)
        value.append(value[0])
    attr2value = np.array(attr2value)
    max_value = np.max(attr2value, axis=0)
    min_value = np.min(attr2value, axis=0)
    for i, (tracker_name, ret) in enumerate(result.items()):
        value = list(ret.values())
        value.append(value[0])
        value = np.array(value)
        value *= (1 / max_value)
        plt.plot(angles, value, linestyle='-', color=COLOR[i], marker=MARKER_STYLE[i],
                label=tracker_name, linewidth=1.5, markersize=6)

    attrs = ["Overall", "Camera motion",
             "Illumination change","Motion Change",
             "Size change","Occlusion",
             "Unassigned"]
    attr_value = []
    for attr, maxv, minv in zip(attrs, max_value, min_value):
        attr_value.append(attr + "\n({:.3f},{:.3f})".format(minv, maxv))
    ax.set_thetagrids(angles[:-1] * 180/np.pi, attr_value)
    ax.spines['polar'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.07), frameon=False, ncol=5)
    ax.grid(b=False)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    plt.show()

if __name__ == '__main__':
    result = pickle.load(open("../../result.pkl", 'rb'))
    draw_eao(result)
