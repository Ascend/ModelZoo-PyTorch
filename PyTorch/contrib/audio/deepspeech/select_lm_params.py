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

import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Select the best parameters based on the WER')
parser.add_argument('--input-path', type=str, help='Output json file from search_lm_params')
args = parser.parse_args()

with open(args.input_path) as f:
    results = json.load(f)

min_results = min(results, key=lambda x: x[2])  # Find the minimum WER (alpha, beta, WER, CER)
print("Alpha: %f \nBeta: %f \nWER: %f\nCER: %f" % tuple(min_results))

alpha, beta, *_ = list(zip(*results))
alpha = np.array(sorted(list(set(alpha))))
beta = np.array(sorted(list(set(beta))))
X, Y = np.meshgrid(alpha, beta)
results = {(a, b): (w, c) for a, b, w, c in results}
WER = np.array([[results[(a, b)][0] for a in alpha] for b in beta])

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(
    X,
    Y,
    WER,
    cmap=matplotlib.cm.rainbow,
    linewidth=0,
    antialiased=False
)
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('WER')
ax.set_zlim(5., 101.)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
