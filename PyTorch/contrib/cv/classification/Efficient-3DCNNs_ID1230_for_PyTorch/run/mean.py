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
def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value]


def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]
