# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
from tqdm import tqdm
import time
import argparse

import numpy as np
import torch
from torch.utils import data
import torch_aie
import mmcv
from mmcv.runner import load_checkpoint
from mmaction.models import build_model


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output

def parse_args():
    parser = argparse.ArgumentParser(description="C3D Sample")
    parser.add_argument('--config', help='please set test config file path', required=True)
    parser.add_argument('--checkpoint', help='please set checkpoint file',  required=True)
    parser.add_argument('--is-localizer', action='store_true', help='whether it is a localizer')
    parser.add_argument('--shape', type=int, nargs="+", default = [1, 10, 3, 16, 112, 112], help='input video size')
    parser.add_argument('--device_id', type=int, default = 0, help="please set device id")
    parser.add_argument('--prep_data', type=str, required=True, help="please set prepocess data dir")
    parser.add_argument('--batch_size', type=int, default = 1, help="please set batch_size")
    parser.add_argument('--label_file', type=str, help="please set batch_size", required=True)
    args = parser.parse_args()
    return args


class CustomDataset(data.Dataset):
    def __init__(self, args):
        self.prep_data_list = os.listdir(args.prep_data)
        self.args = args

    def __getitem__(self, index):
        bin_file = self.prep_data_list[index - 1]
        input_tensor = torch.from_numpy(np.fromfile(os.path.join(self.args.prep_data, bin_file), dtype=np.float32)) \
            .reshape([10, 3, 16, 112, 112])
        return input_tensor, bin_file

    def __len__(self):
        return len(self.prep_data_list)


class TestC3D:

    def __init__(self, args):
        self.args = args
        self.device_name = "npu:" + str(args.device_id)
        self.stream = torch_aie.npu.Stream(self.device_name)
        self.src_nn_module = None
        self.compiled_torch_aie_bs1 = None
        self.bs1_shape = [1, 10, 3, 16, 112, 112]
        self.labels_dict = None

    def load_nn_module(self):
        print("start load C3D nn module.")
        cfg = mmcv.Config.fromfile(self.args.config)
        if not args.is_localizer:
            cfg.model.backbone.pretrained = None
        # build the model
        model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        model = _convert_batchnorm(model)

        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        elif hasattr(model, '_forward') and args.is_localizer:
            model.forward = model._forward
        else:
            raise NotImplementedError(
                'Please implement the forward method for exporting.')
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        self.src_nn_module = model.eval()
        print("load C3D nn module success.")

    def compile(self, batch_size = 1):
        print("start compile model.")
        if not self.src_nn_module:
            self.load_nn_module()
        if self.compiled_torch_aie_bs1 is not None and batch_size == 1:
            print("compile model success, bs is,", batch_size)
            return self.compiled_torch_aie_bs1
        elif self.src_nn_module is not None:
            input_shape = [batch_size, 10, 3, 16, 112, 112]

            print("start trace model.")
            traced_model = torch.jit.trace(self.src_nn_module, torch.randn(self.bs1_shape))
            print("trace model success.")

            compiled_model = torch_aie.compile(traced_model, inputs = [torch_aie.Input(input_shape)])
            print("compile model success.")
            if batch_size == 1:
                self.compiled_torch_aie_bs1 = compiled_model
            return compiled_model
        else:
            print("compile model failed.")

    def process_label(self):
        print("start generate labels.")
        label = dict()
        with open(self.args.label_file, 'r') as f:
            x = f.readlines()
        for i in range(len(x)):
            class_name = x[i].split(' ')[0].split('/')[1]
            class_idx = x[i].split(' ')[2].replace('\n', '').replace('\r', '')
            label[class_name] = class_idx
        self.labels_dict = label
        print("generate labels success.")

    def cal_accuracy(self):
        print("start calculate accuracy.")
        if self.labels_dict is None:
            self.process_label()
        num_correct_top1 = 0
        num_total = len(os.listdir(self.args.prep_data))
        if self.compiled_torch_aie_bs1 is None:
            self.compile(batch_size=1)
        for file in tqdm(os.listdir(self.args.prep_data)):
            if file.endswith(".bin"):
                input_tensor = torch.from_numpy(np.fromfile(os.path.join(self.args.prep_data, file), dtype=np.float32)) \
                    .reshape(self.bs1_shape).to(self.device_name)
                with torch_aie.npu.stream(self.stream):
                    torch_aie_ret = self.compiled_torch_aie_bs1(input_tensor)
                    self.stream.synchronize()
                    torch_aie_ret = torch_aie_ret[0].to("cpu")
                    pre_cls = torch.argmax(torch_aie_ret.mean(dim=0)).numpy()
                    if self.labels_dict[file.replace('.bin', '')] == str(pre_cls):
                        num_correct_top1 += 1
        top1_acc = num_correct_top1 / num_total
        result_dict = {"top1_acc": top1_acc}
        print("calculate acc done,", result_dict)

    def cal_qps(self, model, batch_size, warm_up=5):
        print("start caculate qps.")
        cost_steps = []
        dataset = CustomDataset(self.args)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, drop_last=True, shuffle=False)
        for index, (data, _) in tqdm(enumerate(dataloader)):
            data = data.to(self.device_name)
            with torch_aie.npu.stream(self.stream):
                ts = time.time()
                pre_hr = model(data)
                self.stream.synchronize()
                te = time.time()
                cost_steps.append(te - ts)
        print("QPS is {}, bs is {}".format(
            batch_size * (len(cost_steps) - warm_up) / sum(cost_steps[warm_up:-1]), batch_size))

if __name__ == '__main__':
    args = parse_args()
    torch_aie.set_device(args.device_id)
    c3d_sample = TestC3D(args)
    model = c3d_sample.compile(args.batch_size)
    c3d_sample.cal_accuracy()
    c3d_sample.cal_qps(model, args.batch_size)