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

import os
import stat
import logging
from typing import Any, List

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
logger = logging.getLogger(__name__)


class DynamicCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataset: Any = None, cache_file: str = None):
        super(DynamicCalibrator, self).__init__()
        self.cache_file = cache_file
        self.dataset = dataset
        self.batch_size = 1
        self.cal_dataset_cnt = len(self.dataset)
        self.current_index = 0

        self.cuda_dataset = []
        for data_list, _ in self.dataset:
            tmp = []
            for data in data_list:
                data_allocation = cuda.mem_alloc(data.nbytes)
                cuda.memcpy_htod(data_allocation, np.ascontiguousarray(data))
                tmp.append(data_allocation)
            self.cuda_dataset.append(tmp)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: Any) -> List:
        if self.current_index >= self.cal_dataset_cnt:
            return []
        ret = self.cuda_dataset[self.current_index]
        self.current_index += 1
        return ret

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        return False

    def write_calibration_cache(self, cache: str):
        return True


class Builder:
    def __init__(
            self, model_path: str, trt_path: str, input_shapes: dict,
            precision: str
    ):
        self.model_path = model_path
        self.trt_path = trt_path
        self.input_shapes = input_shapes
        self.precision = precision

        self.dataset = None

    def set_calibration_dataset(self, dataset):
        self.dataset = dataset

    def _get_opti_profile(self, builder, network):
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            min_shape = trt.Dims(
                self.input_shapes.get(input_name)["min_shapes"])
            opt_shape = trt.Dims(
                self.input_shapes.get(input_name)["opt_shapes"])
            max_shape = trt.Dims(
                self.input_shapes.get(input_name)["max_shapes"])
            profile.set_shape(
                input_name, min=min_shape, opt=opt_shape, max=max_shape
            )
        return profile

    def _set_precision(self, config, builder, network):
        config.set_preview_feature(
            trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True
        )
        config.set_flag(trt.BuilderFlag.FP16)
        if self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            if self.dataset is None:
                self._set_tensor_dynamic_range(network)
            else:
                calib_profile = self._get_opti_profile(builder, network)
                config.set_calibration_profile(calib_profile)
                calibrator = DynamicCalibrator(self.dataset)
                config.int8_calibrator = calibrator

    def build(self) -> bool:
        if not os.path.exists(self.model_path):
            logger.error(f"File {self.model_path} "
                         f"is not exited, please check it .")
            return False
        if os.path.exists(self.trt_path):
            logger.warning(
                f"File {self.trt_path} is already existed, skip build it .")
            return True
        os.makedirs(os.path.dirname(self.trt_path), exist_ok=True)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = (1 << 30) * 20

            logger.info(f"Build model from {self.model_path}")
            with open(self.model_path, "rb") as model:
                if not parser.parse(model.read()):
                    logger.error(f"Failed to parse the file {self.model_path}")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))

            self._set_precision(config, builder, network)
            profile = self._get_opti_profile(builder, network)
            config.add_optimization_profile(profile)
            plan = builder.build_serialized_network(network, config)

            if plan is None:
                logger.error("engine build filed")
                return False
            logger.info("Completed creating Engine")
            with os.fdopen(os.open('label.txt', os.O_WRONLY, stat.S_IWUSR), 'w') as f:
                f.write(plan)
            return True

    @staticmethod
    def _set_tensor_dynamic_range(network, in_range=2.0, out_range=4.0):
        def set_input():
            for input_index in range(layer.num_inputs):
                layer_input = layer.get_input(input_index)
                if layer_input and not layer_input.dynamic_range:
                    dyn_range = (
                        out_range
                        if layer.type == trt.LayerType.CONCATENATION
                        else in_range
                    )
                    if not layer_input.set_dynamic_range(-dyn_range, dyn_range):
                        return False
            return True

        def set_output():
            for output_index in range(layer.num_outputs):
                layer_output = layer.get_output(output_index)
                if layer_output and not layer_output.dynamic_range:
                    dyn_range = (
                        in_range
                        if layer.type == trt.LayerType.POOLING
                        else out_range
                    )
                    if not layer_output.set_dynamic_range(
                            -dyn_range, dyn_range
                    ):
                        return False
            return True

        for layer_index in range(network.num_layers):
            layer = network.get_layer(layer_index)
            if not set_input() or not set_output():
                return False
        return True
