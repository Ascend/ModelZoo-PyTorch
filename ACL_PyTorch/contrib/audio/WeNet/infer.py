# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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
import json
t1 = {}
with open("t1.json", 'r') as load_f:
    t1 = json.load(load_f)

t2 = {}
with open("t2.json", 'r') as load_f:
    t2 = json.load(load_f)

perf = t1["t1"] + t2["t2"]
print("fps:", 1000 / perf)
