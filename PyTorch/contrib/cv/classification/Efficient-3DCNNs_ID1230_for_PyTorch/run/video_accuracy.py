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
from run.eval_ucf101 import UCFclassification

ucf_classification = UCFclassification('../annotation_UCF101/ucf101_01.json',
                                       '../results/val.json',
                                       subset='validation', top_k=1)
ucf_classification.evaluate()

print("test top1 acc:"+str(ucf_classification.hit_at_k))
