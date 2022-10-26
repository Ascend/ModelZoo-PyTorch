# Copyright 2022 Huawei Technologies Co., Ltd
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

import copy
import numpy as np


class MPIIEval:
    """
    eval for MPII dataset with PCK
    """
    template = {
        "all": {
            "total": 0,
            "ankle": 0,
            "knee": 0,
            "hip": 0,
            "pelvis": 0,
            "thorax": 0,
            "neck": 0,
            "head": 0,
            "wrist": 0,
            "elbow": 0,
            "shoulder": 0,
        },
        "visible": {
            "total": 0,
            "ankle": 0,
            "knee": 0,
            "hip": 0,
            "pelvis": 0,
            "thorax": 0,
            "neck": 0,
            "head": 0,
            "wrist": 0,
            "elbow": 0,
            "shoulder": 0,
        },
        "not visible": {
            "total": 0,
            "ankle": 0,
            "knee": 0,
            "hip": 0,
            "pelvis": 0,
            "thorax": 0,
            "neck": 0,
            "head": 0,
            "wrist": 0,
            "elbow": 0,
            "shoulder": 0,
        },
    }

    joint_map = [
        "ankle",
        "knee",
        "hip",
        "hip",
        "knee",
        "ankle",
        "pelvis",
        "thorax",
        "neck",
        "head",
        "wrist",
        "elbow",
        "shoulder",
        "shoulder",
        "elbow",
        "wrist",
    ]

    def __init__(self):
        self.correct_valid = copy.deepcopy(self.template)
        self.count_valid = copy.deepcopy(self.template)

    def eval(self, pred, gt, normalizing, bound=0.5):
        """
        use PCK with threshold of .5 of normalized distance (presumably head size)
        """
        for p, g, normalize in zip(pred, gt, normalizing):
            for j in range(g.shape[1]):
                vis = "visible"
                if g[0, j, 0] == 0: # Not in image
                    continue
                if g[0, j, 2] == 0:
                    vis = "not visible"
                joint = self.joint_map[j]

                self.count_valid["all"]["total"] += 1
                self.count_valid["all"][joint] += 1
                self.count_valid[vis]["total"] += 1
                self.count_valid[vis][joint] += 1

                error = np.linalg.norm(p[0]["keypoints"][j, :2] - g[0, j, :2]) / normalize
  
                if bound > error:
                    self.correct_valid["all"]["total"] += 1
                    self.correct_valid["all"][joint] += 1
                    self.correct_valid[vis]["total"] += 1
                    self.correct_valid[vis][joint] += 1
        self.output_result(bound)

    def output_result(self, bound):
        """
        output split via valid
        """
        for k in self.correct_valid:
            print(k, ":")
            for key in self.correct_valid[k]:
                print(
                    "Val PCK @,",
                    bound,
                    ",",
                    key,
                    ":",
                    round(self.correct_valid[k][key] / max(self.count_valid[k][key], 1), 3),
                    ", count:",
                    self.count_valid[k][key],
                )
            print("\n")