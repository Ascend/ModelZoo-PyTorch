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

import numpy as np

if __name__ == '__main__':
    f_mx = np.loadtxt("../mxbase/test_result.txt")
    f_sdk = np.loadtxt("../sdk/test_result.txt")
    f_mx = f_mx.reshape((520, 224, 224))
    f_sdk = f_sdk.reshape((520, 224, 224))
    error = np.sum(f_mx - f_sdk)
    print("the inference error between mxbase and sdk", error)
