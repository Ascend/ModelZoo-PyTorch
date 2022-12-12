# Copyright 2022 Huawei Technologies Co., Ltd
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
import torch.nn.functional as F
import torch
from mmcv import Config
from mmaction.datasets import build_dataset, build_dataloader
import aclruntime
import numpy as np

class InferSession:
    def __init__(self, device_id: int, model_path: str, acl_json_path: str = None, debug: bool = False, loop: int = 1):
        """
        init InferSession

        Args:
            device_id: device id for npu device
            model_path: om model path to load
            acl_json_path: set acl_json_path to enable profiling or dump function
            debug: enable debug log.  Default: False
            loop: loop count for one inference. Default: 1
        """
        self.device_id = device_id
        self.model_path = model_path
        self.loop = loop
        options = aclruntime.session_options()
        if acl_json_path is not None:
            options.acl_json_path = acl_json_path
        options.log_level = 1 if debug == True else 2
        options.loop = self.loop
        self.session = aclruntime.InferenceSession(self.model_path, self.device_id, options)
        self.outputs_names = [meta.name for meta in self.session.get_outputs()]

    def create_tensor_from_arrays_to_device(self, arrays):
        tensor = aclruntime.Tensor(arrays)
        tensor.to_device(self.device_id)
        return tensor

    def convert_tensors_to_host(self, tensors):
        for tensor in tensors:
            tensor.to_host()

    def convert_tensors_to_arrays(self, tensors):
        arrays = []
        for tensor in tensors:
            # convert acltensor to numpy array
            arrays.append(np.array(tensor))
        return arrays

    def run(self, feeds, out_array=False):
        inputs = feeds
        outputs = self.session.run(self.outputs_names, inputs)
        if out_array == True:
            # convert to host tensor
            self.convert_tensors_to_host(outputs)
            # convert tensor to narray
            return self.convert_tensors_to_arrays(outputs)
        else:
            return outputs

def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--model_path',
        default=None,
        help='the path of om model')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" ')
    args = parser.parse_args()
    return args



def infer_session_simple(model_path,data_loader):
    session = InferSession(device_id = 0, model_path = model_path)
    
    output = []
    for i , data in enumerate(data_loader,1):
        batch_bin = data['imgs'].cpu().numpy()
        tensor = session.create_tensor_from_arrays_to_device(batch_bin)
        feeds = [ tensor ]
        outputs = session.run(feeds, out_array=True)
        outputs = torch.from_numpy(outputs[0])
        outputs = F.softmax(outputs, dim=1).mean(dim=0)
        outputs = outputs.numpy()
        output.extend([outputs])
        
    eval_res = dataset.evaluate(output, **eval_config)
    for name, val in eval_res.items():
        print(f'{name}: {val:.04f}')

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    eval_config = cfg.get('eval_config', {})
    eval_config = Config._merge_a_into_b(dict(metrics=args.eval), eval_config)
    dataset_type = cfg.data.test.type
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)
    infer_session_simple(args.model_path,data_loader)

