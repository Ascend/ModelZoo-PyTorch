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

import logging
import os
import time

from typing import List, Any, Dict

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
logger = logging.getLogger(__name__)


class MirroredBuffer:
    def __init__(
        self,
        host_mem: Any,
        device_mem: Any,
        size: Any,
        dtype: Any,
        shape: Any,
        index: Any,
    ) -> None:
        self.size = size
        self.dtype = np.dtype(dtype)
        self.host = host_mem
        self.device = device_mem
        self.shape = shape
        self.index = index


class Infer:
    def __init__(self, trt_path: str, max_shape: List[List], warm_up_iteration=0):
        self.trt_path = trt_path
        self.max_shape = max_shape
        self.warm_up_iteration = warm_up_iteration

        self.pre_input = None
        self.context = None
        self.engine = None

        self.inputs = []
        self.outputs = {}
        self.bindings = []

        self.results = []
        self.h2d_times = []
        self.execute_times = []
        self.d2h_times = []
        self.set_dynamic_times = []
        self.e2e_start_time = None
        self.e2e_end_time = None
        self.host_wall_time = None

        self._load()

    def _set_input_shape(self, input_shapes):
        for index, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                origin_shape = self.context.get_binding_shape(index)
                for i, value in enumerate(input_shapes[index]):
                    origin_shape[i] = value
                self.context.set_binding_shape(index, origin_shape)
            else:
                shape = list(self.context.get_binding_shape(index))
                input_shapes.append(shape)

    def _allocate_buffers(self, input_shapes):
        self._set_input_shape(input_shapes)
        inputs = []
        outputs = {}
        bindings = []
        for index, binding in enumerate(self.engine):
            shape = input_shapes[index]
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            buffer = MirroredBuffer(
                host_mem, device_mem, size, dtype, shape, index
            )
            if self.engine.binding_is_input(binding):
                inputs.append(buffer)
            else:
                outputs[binding] = buffer
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings

    def _set_dynamic_input(self, input_shapes):
        start = time.time()
        self.context.active_optimization_profile = 0
        self._set_input_shape(input_shapes)
        end = time.time()
        self.set_dynamic_times.append((end - start) * 1000)

    def _load(self):
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        if os.path.exists(self.trt_path):
            logger.info(f"Load engine from file {self.trt_path}")
            with open(self.trt_path, "rb") as f, trt.Runtime(
                TRT_LOGGER
            ) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()

                self._allocate_buffers(self.max_shape)
        else:
            raise FileNotFoundError(f"{self.trt_path} not exit")
    @classmethod
    def _contiguous_data(self, dataloader):
        ret = []
        for datas in dataloader:
            data_tmp = [np.ascontiguousarray(i) for i in datas]
            ret.append(data_tmp)
        return ret

    def _fill_data(self, datas):
        for data, inp in zip(datas, self.inputs):
            inp.host = data

    def _memcpy_h2d(self):
        start = time.time()
        for inp in self.inputs:
            cuda.memcpy_htod(inp.device, inp.host)
        end = time.time()
        self.h2d_times.append((end - start) * 1000)

    def _compute(self):
        start = time.time()
        self.context.execute_v2(bindings=self.bindings)
        end = time.time()
        self.execute_times.append((end - start) * 1000)

    def _memcpy_d2h(self):
        start = time.time()
        for out in self.outputs.values():
            cuda.memcpy_dtoh(out.host, out.device)
        end = time.time()
        self.d2h_times.append((end - start) * 1000)

    def _save_output(self):
        results = {}
        for binding, out in self.outputs.items():
            data = np.array(
                out.host[
                    0 : np.array(
                        self.context.get_binding_shape(out.index)
                    ).prod()
                ].reshape(self.context.get_binding_shape(out.index))
            )
            results[binding] = data
        return results

    def warmup(self, iterations: int):
        logger.info(f"begin to warm up {iterations} iterations")
        for _ in range(iterations):
            self._compute()
        self.execute_times.clear()
        logger.info(f"warm up finish.")

    def infer(self, data: List) -> Dict[str, np.array]:
        input_shape = [i.shape for i in data]
        if self.pre_input is None or self.pre_input != input_shape:
            self._set_dynamic_input(input_shape)
            self.pre_input = input_shape
        self._fill_data(data)
        self._memcpy_h2d()
        self._compute()
        self._memcpy_d2h()

        result = self._save_output()

        return result
