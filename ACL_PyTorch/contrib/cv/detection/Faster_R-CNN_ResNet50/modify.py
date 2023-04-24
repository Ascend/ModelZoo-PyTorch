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

from auto_optimizer import OnnxGraph

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="./faster_rcnn_r50_fpn.onnx")
parser.add_argument('--output', type=str, default="./faster_rcnn_r50_fpn_m.onnx")

if __name__ == '__main__':
	opts = parser.parse_args()
	model=OnnxGraph.parse(opts.model)
	model.remove('Split_434')
	model.save(opts.output)