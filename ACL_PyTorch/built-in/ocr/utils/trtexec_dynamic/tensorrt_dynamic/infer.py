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

# coding=utf-8

import logging
import os
import time

from typing import List, Any, Dict

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from pycuda.tools import clear_context_caches

from polygraphy import cuda as pgcuda
from polygraphy import util

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
logger = logging.getLogger(__name__)


def _make_output_allocator():
    class OutputAllocator(trt.IOutputAllocator):
        def __init__(self):
            trt.IOutputAllocator.__init__(self)
            self.buffers = {}
            self.shapes = {}

        def reallocate_output(self, tensor_name, memory, size, alignment):
            shape = (size,)
            if tensor_name not in self.buffers:
                self.buffers[tensor_name] = pgcuda.DeviceArray.raw(shape)
            else:
                self.buffers[tensor_name].resize(shape)
            logger.debug(
                f"Reallocated output tensor: {tensor_name} to: {self.buffers[tensor_name]}, size: {size}")
            return self.buffers[tensor_name].ptr

        def notify_shape(self, tensor_name, shape):
            self.shapes[tensor_name] = tuple(shape)

    return OutputAllocator()


def np_dtype_from_trt(trt_dtype):
    # trt.nptype uses NumPy, so to make autoinstall work, we need to trigger it before that.
    return np.dtype(trt.nptype(trt_dtype))


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
    def __init__(self, trt_path: str, warm_up_iteration=0, device_id=0):
        self.trt_path = trt_path
        self.warm_up_iteration = warm_up_iteration

        self.pre_input = None
        self.context = None
        self.engine = None
        self.allocator = None
        self.stream = None

        self.inputs = []
        self.outputs = {}
        self.bindings = []
        self.output_names = []

        self.results = []
        self.h2d_times = []
        self.execute_times = []
        self.d2h_times = []
        self.set_dynamic_times = []
        self.e2e_start_time = None
        self.e2e_end_time = None
        self.host_wall_time = None

        self.make_device_context(device_id)

        self._load()

    def __del__(self):
        self._finalize()

    def _finalize(self) -> None:
        if self.device_context:
            self.device_context.pop()
        self.device_context = None

        clear_context_caches()

    def make_device_context(self, device_id):
        self.device_context = cuda.Device(device_id).make_context()

    def _set_input_shape(self, input_shapes):
        for index, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                origin_shape = self.context.get_binding_shape(index)
                for i, value in enumerate(input_shapes[index]):
                    origin_shape[i] = value
                self.context.set_input_shape(binding, origin_shape)
            else:
                shape = list(self.context.get_tensor_shape(binding))
                input_shapes.append(shape)

    def _allocate_buffers(self, input_shapes):
        self.stream = pgcuda.Stream()
        self.allocator = _make_output_allocator()

        index = 0
        if not self.context.set_optimization_profile_async(index,
                                                           self.stream.ptr):
            logger.debug(f"Failed to set optimization profile to: {index}")

        inputs = []
        outputs = {}
        bindings = []
        output_names = []

        for index, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                shape = input_shapes[index]
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                buffer = MirroredBuffer(
                    host_mem, device_mem, size, dtype, shape, index
                )
                inputs.append(buffer)
                if not self.context.set_tensor_address(binding,
                                                       int(device_mem)):
                    logger.debug(f"set {binding} address failed!")
            else:
                outputs[binding] = np.empty(shape=tuple(), dtype=np.byte)
                output_names.append(binding)
                if self.context.set_output_allocator(binding, self.allocator):
                    logger.debug(f"set {binding} allocator success")
                else:
                    logger.debug(f"set {binding} allocator failed!")

        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.output_names = output_names

        self._set_input_shape(input_shapes)

    def _set_dynamic_input(self, input_shapes):
        start = time.time()
        self._set_input_shape(input_shapes)
        end = time.time()
        self.set_dynamic_times.append((end - start) * 1000)

    def _get_max_shapes(self):
        max_shapes = []
        for index, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                shapes = self.engine.get_profile_shape(0, index)
                max_shapes.append(shapes[2])
        return max_shapes

    def _load(self):
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        if os.path.exists(self.trt_path):
            logger.debug(f"Load engine from file {self.trt_path}")
            with open(self.trt_path, "rb") as f, trt.Runtime(
                    TRT_LOGGER
            ) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
                self.context.active_optimization_profile = 0
                self._allocate_buffers(self._get_max_shapes())
        else:
            raise FileNotFoundError(f"{self.trt_path} not exit")

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
        if not self.context.execute_async_v3(self.stream.ptr):
            return False
        self.stream.synchronize()
        end = time.time()
        self.execute_times.append((end - start) * 1000)
        return True

    def _memcpy_d2h(self):
        outputs = {}
        start = time.time()
        for name in self.output_names:
            raw_array = self.allocator.buffers[name]
            shape = self.allocator.shapes[name]
            dtype = np_dtype_from_trt(
                self.engine.get_tensor_dtype(name))
            nbytes = raw_array.nbytes
            self.outputs[name] = util.resize_buffer(
                self.outputs[name], (nbytes,))
            raw_array.view(shape=(nbytes,)).copy_to(
                self.outputs[name], stream=self.stream)
            raw_array = self.outputs[name]

            array = raw_array.view(dtype)[0: np.prod(shape)].reshape(shape)
            outputs[name] = array
        end = time.time()
        self.d2h_times.append((end - start) * 1000)
        return outputs

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
        if not self._compute():
            return None

        result = self._memcpy_d2h()

        return result
