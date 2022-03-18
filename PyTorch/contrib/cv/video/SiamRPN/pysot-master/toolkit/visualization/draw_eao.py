# Copyright 2021 Huawei Technologies Co., Ltd
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


import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib import rc
from .draw_utils import COLOR, MARKER_STYLE

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def draw_eao(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=True)

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
             "Illumination change", "Motion Change",
             "Size change", "Occlusion",
             "Unassigned"]
    attr_value = []
    for attr, maxv, minv in zip(attrs, max_value, min_value):
        attr_value.append(attr + "\n({:.3f},{:.3f})".format(minv, maxv))
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, attr_value)
    ax.spines['polar'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=5)
    ax.grid(b=False)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    plt.show()


if __name__ == '__main__':
    result = pickle.load(open("../../result.pkl", 'rb'))
    draw_eao(result)
