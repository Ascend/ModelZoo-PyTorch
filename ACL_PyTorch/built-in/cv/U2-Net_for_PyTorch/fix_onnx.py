# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from magiconnx import OnnxGraph


BS = 1
SHAPE_INFO = {
    'Resize_42': [[BS, 32, 10, 10], [BS, 32, 20, 20]],
    'Resize_64': [[BS, 32, 20, 20], [BS, 32, 40, 40]],
    'Resize_86': [[BS, 32, 40, 40], [BS, 32, 80, 80]],
    'Resize_108': [[BS, 32, 80, 80], [BS, 32, 160, 160]],
    'Resize_130': [[BS, 32, 160, 160], [BS, 32, 320, 320]],
    'Resize_175': [[BS, 32, 10, 10], [BS, 32, 20, 20]],
    'Resize_197': [[BS, 32, 20, 20], [BS, 32, 40, 40]],
    'Resize_219': [[BS, 32, 40, 40], [BS, 32, 80, 80]],
    'Resize_241': [[BS, 32, 80, 80], [BS, 32, 160, 160]],
    'Resize_283': [[BS, 64, 10, 10], [BS, 64, 20, 20]],
    'Resize_305': [[BS, 64, 20, 20], [BS, 64, 40, 40]],
    'Resize_327': [[BS, 64, 40, 40], [BS, 64, 80, 80]],
    'Resize_366': [[BS, 128, 10, 10], [BS, 128, 20, 20]],
    'Resize_388': [[BS, 128, 20, 20], [BS, 128, 40, 40]],
    'Resize_453': [[BS, 512, 10, 10], [BS, 512, 20, 20]],
    'Resize_493': [[BS, 512, 20, 20], [BS, 512, 40, 40]],
    'Resize_528': [[BS, 128, 10, 10], [BS, 128, 20, 20]],
    'Resize_550': [[BS, 128, 20, 20], [BS, 128, 40, 40]],
    'Resize_573': [[BS, 256, 40, 40], [BS, 256, 80, 80]],
    'Resize_611': [[BS, 64, 10, 10], [BS, 64, 20, 20]],
    'Resize_633': [[BS, 64, 20, 20], [BS, 64, 40, 40]],
    'Resize_655': [[BS, 64, 40, 40], [BS, 64, 80, 80]],
    'Resize_678': [[BS, 128, 80, 80], [BS, 128, 160, 160]],
    'Resize_719': [[BS, 32, 10, 10], [BS, 32, 20, 20]],
    'Resize_741': [[BS, 32, 20, 20], [BS, 32, 40, 40]],
    'Resize_763': [[BS, 32, 40, 40], [BS, 32, 80, 80]],
    'Resize_785': [[BS, 32, 80, 80], [BS, 32, 160, 160]],
    'Resize_808': [[BS, 64, 160, 160], [BS, 64, 320, 320]],
    'Resize_852': [[BS, 16, 10, 10], [BS, 16, 20, 20]],
    'Resize_874': [[BS, 16, 20, 20], [BS, 16, 40, 40]],
    'Resize_896': [[BS, 16, 40, 40], [BS, 16, 80, 80]],
    'Resize_918': [[BS, 16, 80, 80], [BS, 16, 160, 160]],
    'Resize_940': [[BS, 16, 160, 160], [BS, 16, 320, 320]],
    'Resize_1045': [[BS, 1, 10, 10], [BS, 1, 320, 320]],
    'Resize_1025': [[BS, 1, 20, 20], [BS, 1, 320, 320]],
    'Resize_1005': [[BS, 1, 40, 40], [BS, 1, 320, 320]],
    'Resize_985': [[BS, 1, 80, 80], [BS, 1, 320, 320]],
    'Resize_965': [[BS, 1, 160, 160], [BS, 1, 320, 320]]
}


def fix_resizev2():
    resize_nodes = onnx_graph.get_nodes('Resize')

    num_fix = 0
    for node in resize_nodes:
        input_shape, output_shape = SHAPE_INFO[node.name]
        if max(input_shape) >= 160 or max(output_shape) >= 160:
            print("{}:{}/{}".format(node.name, input_shape, output_shape))
        else:
            continue
        node['coordinate_transformation_mode'] = 'asymmetric'
        node['mode'] = 'nearest'
        num_fix += 1
    print("num of fix: {}/{}".format(num_fix, len(resize_nodes)))


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    onnx_graph = OnnxGraph(input_path)
    fix_resizev2()
    onnx_graph.save(output_path)
