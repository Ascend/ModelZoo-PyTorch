# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
import os
import time
from collections import OrderedDict
import torch
import torch_npu
import torch_aie
from torch_aie import _enums

from tqdm import tqdm
import numpy as np
from concern.config import Configurable, Config

AIE_MODEL_DEFAULT_NAME = 'aie_model_bs'


def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for inference')
    parser.add_argument('--device', type=int, default=0,
                        help='npu device id for inference')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--result_dir', type=str,
                        default='./results/', help='path to save results')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--verbose', action='store_true',
                        help='show verbose info')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--trace_compile', action='store_true',
                        help='trace and compile aie model if true')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.set_defaults(debug=False, verbose=False)

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Eval(experiment, experiment_args, cmd=args,
         verbose=args['verbose']).eval(args['trace_compile'], args['device'], args['batch_size'])


def proc_nodes_modile(checkpoint):
    # need to modify states in order for the model to work
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def compile_aie_model(torch_model, batch_size):
    accept_size = [batch_size, 3, 736, 1280]
    dummy_input = torch.rand(accept_size) / 2
    with torch.no_grad():
        jit_model = torch.jit.trace(torch_model, dummy_input)
    aie_input_spec = [
        torch_aie.Input(accept_size, dtype=torch_aie.dtype.FLOAT16),  # Static NCHW input shape for input #1
    ]
    aie_model = torch_aie.compile(
        jit_model,
        inputs=aie_input_spec,
        precision_policy=_enums.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        require_full_compilation=False,
        allow_tensor_replace_int=False,
        min_block_size=3,
        torch_executed_ops=[],
        soc_version="Ascend310P3",
        optimization_level=0)
    aie_model.save(AIE_MODEL_DEFAULT_NAME + str(batch_size) + ".pt")
    return aie_model


class Eval:
    def __init__(self, experiment, args, cmd, verbose=False):
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.data_loaders = experiment.evaluation.data_loaders
        self.args = cmd
        self.logger = experiment.logger
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = cmd.get(
            'resume', os.path.join(
                self.logger.save_dir(model_saver.dir_path),
                'final'))
        self.verbose = verbose

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            self.logger.warning("Checkpoint not found: " + path)
            return
        self.logger.info("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        states = proc_nodes_modile(states)
        model.load_state_dict(states, strict=False)
        self.logger.info("Resumed from " + path)

    def report_speed(self, aie_model, batch, times=100):
        data = {k: v[0:1]for k, v in batch.items()}
        start = time.time()
        for _ in range(times):
            pred = aie_model(batch['image'])
        time_cost = (time.time() - start) / times
        for _ in range(times):
            output = self.structure.representer.represent(
                batch, pred, is_output_polygon=False)
        self.logger.info('Inference speed: %fms, FPS: %f' % (
            time_cost * 1000, 1 / time_cost))

        return time_cost

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + \
                filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(
                self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i, :, :].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")

    def eval(self, trace_compile=True, device=0, batch_size=1):
        torch.npu.set_device(device)
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        model.eval()

        if trace_compile:
            aie_model = compile_aie_model(model, batch_size)
        else:
            aie_model = torch.jit.load(
                AIE_MODEL_DEFAULT_NAME + str(batch_size) + ".pt")

        with torch.no_grad():
            for _, data_loader in self.data_loaders.items():
                # Single stream inference
                # warm up
                random_input = torch.randn(
                    [batch_size, 3, 736, 1280], dtype=torch.half, device=torch_npu.npu.current_device())
                pred = aie_model(random_input)
                self.logger.info(
                    "Single stream inference warm up done, begin inference...")
                raw_metrics = []
                model_time = []
                all_start = time.time()
                for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    batch_inputs = batch['image'].npu().to(torch.half)
                    torch.npu.synchronize()
                    start = time.time()
                    pred = aie_model(batch_inputs)
                    torch.npu.synchronize()
                    model_time.append(time.time() - start)
                    output = self.structure.representer.represent(
                        batch, pred.cpu(), is_output_polygon=self.args['polygon'])
                    if not os.path.isdir(self.args['result_dir']):
                        os.mkdir(self.args['result_dir'])
                    self.format_output(batch, output)
                    raw_metric = self.structure.measurer.validate_measure(
                        batch, output, is_output_polygon=self.args['polygon'], box_thresh=self.args['box_thresh'])
                    raw_metrics.append(raw_metric)
                all_end = time.time()
                metrics = self.structure.measurer.gather_measure(
                    raw_metrics, self.logger)
                for key, metric in metrics.items():
                    self.logger.info('%s : %f (%d)' %
                                     (key, metric.avg, metric.count))
                self.logger.info("Single_stream_forward-QPS = %f" %
                                 (batch_size * len(model_time) / sum(model_time)))
                self.logger.info("Single_stream_e2e-QPS(including postprocess) = %f" %
                                 (batch_size * len(model_time) / (all_end - all_start)))

                # Multi stream inference
                # Using last batch for inference performance test, loop N times
                aie_model_npu = aie_model.npu()
                # create stream
                stream_h2d = torch_npu.npu.Stream()
                stream_forward = torch_npu.npu.Stream()
                stream_d2h = torch_npu.npu.Stream()
                input_npus = []  # queue, length=3
                output_npus = []
                output_cpu = None
                # warm up
                random_input_npu = torch.randn(
                    [batch_size, 3, 736, 1280], device=torch_npu.npu.current_device(), dtype=torch.half)
                for j in range(3):
                    random_output_npu = aie_model_npu(random_input_npu)
                    output_cpu = random_output_npu.cpu()
                    input_npus.append(random_input_npu)
                    output_npus.append(random_output_npu)
                # clear 3 streams
                stream_h2d.synchronize()
                stream_forward.synchronize()
                stream_d2h.synchronize()
                self.logger.info(
                    'Multi stream inference warm up done, begin inference...')
                # inference
                loop_times = int(500 / batch_size)
                time_cost = 0
                for j in range(loop_times + 2):
                    loop_start = time.time()
                    # forward
                    if j >= 1 and j < loop_times + 1:
                        with torch_npu.npu.stream(stream_forward):
                            output_npus[(j - 1) %
                                        3] = aie_model_npu(input_npus[(j - 1) % 3])
                    # h2d
                    if j >= 0 and j < loop_times:
                        with torch_npu.npu.stream(stream_h2d):
                            input_npus[j % 3] = batch['image'].to(
                                "npu", non_blocking=True, dtype=torch.half)
                    # d2h
                    if j >= 2:
                        with torch_npu.npu.stream(stream_d2h):
                            output_cpu.copy_(
                                output_npus[(j - 2) % 3], non_blocking=True)
                    # synchronize
                    if j >= 2:
                        stream_d2h.synchronize()
                    stream_h2d.synchronize()
                    if j >= 1 and j < loop_times + 1:
                        stream_forward.synchronize()
                    loop_end = time.time()
                    time_cost += (loop_end - loop_start)
                self.logger.info("Multi_stream_e2e-QPS(no including postprocess) = %f" %
                                 (batch_size * loop_times / time_cost))


if __name__ == '__main__':
    main()
