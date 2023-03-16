# Copyright 2023 Huawei Technologies Co., Ltd
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


import time
import argparse
from pathlib import Path

import tqdm
import numpy as np


class BaseInferenceHelper:
    def __init__(self):
        self.session = None
        self.input_names = None
        self.input_shapes = None
        self.input_dtypes = None
        self.output_names = None
        self.model_type = None

    def init_helper_info(self):
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.input_shapes = [i.shape for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def load_data(self, input_feed):
        paths = [Path(p) for p in input_feed.strip().split(',')]
        assert all(p.is_file() for p in paths) or all(p.is_dir() for p in paths)
        assert len(paths) == len(self.input_names)

        prepare_list = []
        if paths[0].is_file():
            assert all(p.suffix == '.npy' for p in paths), \
                'Only npy files are supported.'
            prepare_list.append(tuple(paths))
        else:
            file_names = set()
            for i, dir_path in enumerate(paths):
                tmp_set = set()
                for p in dir_path.iterdir():
                    assert p.suffix == '.npy', 'Only npy files are supported.'
                    tmp_set.add(p.name)
                if i == 0:
                    file_names.update(tmp_set)
                else:
                    file_names = file_names & tmp_set

            file_names = sorted(file_names)
            for file_name in file_names:
                prepare_list.append([dir_path/file_name for dir_path in paths])

        for item in prepare_list:
            if self.model_type == 'ONNX':
                data = {
                    self.input_names[i]: np.load(path).astype(self.input_dtypes[i])
                    for i, path in enumerate(item)
                }
            else:
                data = [
                    np.load(path).astype(self.input_dtypes[i])
                    for i, path in enumerate(item)
                ]
            yield item[0].name, data

    def inference(self, input_feed=None, output_dir=None, batchsize=1):
        data_iter = self.load_data(input_feed)
        save = False
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save = True

        duration_list = []
        for name, data in tqdm.tqdm(data_iter):
            start_time = time.time()
            outputs = self.single_inference(data)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            duration_list.append(duration)
            if save:
                for i, output in enumerate(outputs):
                    save_path = output_dir/name.replace('.npy', f'_{i}.npy')
                    np.save(save_path, output)

        time_spent = np.sum(duration_list)
        avg_time_without_first = np.mean(duration_list[1:])
        throughput = 1000 * batchsize / avg_time_without_first

        print(f'[INFO] {"-"*22}Performance Summary{"-"*23}')
        print(f'[INFO] Total time: {time_spent:.3f} ms.')
        print(f'[INFO] Average time without first time: {avg_time_without_first:.3f} ms.')
        print(f'[INFO] Throughput: {throughput:.3f} fps.')
        print(f'[INFO] {"-"*64}')


class OmInferenceHelper(BaseInferenceHelper):
    def __init__(self, om_path, device_id=0):
        super(OmInferenceHelper, self).__init__()
        from ais_bench.infer.interface import InferSession
        self.session = InferSession(device_id, om_path)
        self.input_dtypes = [i.datatype.name for i in self.session.get_inputs()]
        self.init_helper_info()

    def single_inference(self, data):
        return self.session.infer(data, 'dymshape')


class OnnxInferenceHelper(BaseInferenceHelper):
    def __init__(self, onnx_path):
        super(OnnxInferenceHelper, self).__init__()
        self.model_type = "ONNX"
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_dtypes = [self.dtype_convert(i.type) 
                             for i in self.session.get_inputs()]
        self.init_helper_info()

    def dtype_convert(self, type_str):
        if type_str == 'tensor(float)':
            return 'float32'
        elif type_str == 'tensor(int32)':
            return 'int32'
        err_msg = f'Please add the convert rule for dtype: {type_str}'
        raise NotImplementedError(err_msg)

    def single_inference(self, data):
        return self.session.run(None, data)


def main():
    parser = argparse.ArgumentParser('Inference for ONNX or OM.')
    parser.add_argument('--model', type=str, required=True, 
                        help='path to the OM model.')
    parser.add_argument('--input', type=str, default=None, 
                        help='path to test data(pickle file).')
    parser.add_argument('--device', default=0, type=int, 
                        help='id number of NPU or GPU.')
    parser.add_argument('--output', default='./output/', type=str, 
                        help='a directory to save result files of inference.')
    args = parser.parse_args()

    if args.model.endswith('.om'):
        helper = OmInferenceHelper(args.model, device_id=args.device)
    elif args.model.endswith('.onnx'):
        helper = OnnxInferenceHelper(args.model)
    else:
        raise Exception(f'Unknown model type: {args.model.rsplit(".")[-1]}')
    helper.inference(input_feed=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
