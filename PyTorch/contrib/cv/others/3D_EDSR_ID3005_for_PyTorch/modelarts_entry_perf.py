# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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

import os
import argparse
import sys

# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/home/ma-user/modelarts/inputs/data_url_0")
parser.add_argument("--train_url", type=str, default="/home/ma-user/modelarts/outputs/train_url_0/")
config = parser.parse_args()

print("[CANN-Modelzoo] code_dir path is [%s]" % (sys.path[0]))
code_dir = sys.path[0]
os.chdir(code_dir)
print("[CANN-Modelzoo] work_dir path is [%s]" % (os.getcwd()))

print("[CANN-Modelzoo] before train - list my run files:")
os.system("ls -al /usr/local/Ascend/ascend-toolkit/")

print("[CANN-Modelzoo] before train - list my dataset files:")
os.system("ls -al %s" % config.data_url)

print("[CANN-Modelzoo] start run train shell")
# 设置sh文件格式为linux可执行
os.system("dos2unix ./test/*")

# 执行train_full_1p.sh或者train_performance_1p.sh，需要用户自己指定
# full和performance的差异，performance只需要执行很少的step，控制在15分钟以内，主要关注性能FPS
os.system("bash ./test/train_performance_1p.sh --data_path=%s --output_path=%s " % (config.data_url, config.train_url))

print("[CANN-Modelzoo] finish run train shell")

# 将当前执行目录所有文件拷贝到obs的output进行备份
print("[CANN-Modelzoo] after train - list my output files:")
os.system("cp -r %s %s " % (code_dir, config.train_url))
os.system("ls -al %s" % config.train_url)
