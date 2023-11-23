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

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from functools import reduce
import ascendie as aie


@dataclass
class RuntimeIOInfo:
    input_shapes: List[tuple]
    input_dtypes: List[type]
    output_shapes: List[tuple]
    output_dtypes: List[type]


class BackgroundRuntime:
    def __init__(
        self,
        device_id: int,
        model_path: str,
        io_info: RuntimeIOInfo,
        engine: aie.Engine,
    ):
        # Create a pipe for process synchronization
        self.sync_pipe, sync_pipe_peer = mp.Pipe(duplex=True)

        # Create shared buffers
        input_spaces = self.create_shared_buffers(io_info.input_shapes,
                                                  io_info.input_dtypes)
        output_spaces = self.create_shared_buffers(io_info.output_shapes,
                                                   io_info.output_dtypes)

        # Build numpy arrays on the shared buffers
        self.input_arrays = [
            np.frombuffer(b, dtype=t).reshape(s) for (b, s, t) in zip(
                input_spaces, io_info.input_shapes, io_info.input_dtypes)
        ]
        self.output_arrays = [
            np.frombuffer(b, dtype=t).reshape(s) for (b, s, t) in zip(
                output_spaces, io_info.output_shapes, io_info.output_dtypes)
        ]

        mp.set_start_method('fork', force=True)
        self.p = mp.Process(target=self.run_infer,
                            args=[
                                sync_pipe_peer, input_spaces, output_spaces,
                                io_info, device_id, model_path, engine
                            ])
        self.p.start()

        # Wait until the sub process is ready
        self.wait()
    
    @staticmethod
    def get_io_info_from_engine(engine: aie.Engine, resolution:List[int]) -> RuntimeIOInfo:
        # Map aclruntime datatype to numpy datatype
        np_types = (np.float32, np.float16, np.int8, np.int32, np.uint8, '',
                    np.int16, np.uint16, np.uint32, np.int64, np.uint64,
                    np.double, np.bool_)
        # Get input shapes and datatypes
        input_num = engine.input_num
        if (engine.get_dynamic_input_type == aie.DynamicInputType.dynamic_dims):
            input_num = engine.input_num - 1
        input_names = [engine.get_input_name(i) for i in range(input_num)]
        input_shapes = []
        for index, name in enumerate(input_names):
            curdims = engine.get_shape(name)
            curshape = []
            if (index == 0 and engine.get_dynamic_input_type == aie.DynamicInputType.dynamic_dims):
                curshape = [1, 4, resolution[0]//8, resolution[1]//8]
            else:
                for i in range(curdims.size):
                    curshape.append(curdims[i])
            input_shapes.append(tuple(curshape))
        input_dtypes = [
            np_types[engine.get_data_type(name)] for name in input_names
        ]

        # Get output shapes and datatypes
        output_num = engine.output_num
        output_names = [engine.get_output_name(i) for i in range(output_num)]
        output_shapes = []
        for index, name in enumerate(output_names):
            curdims = engine.get_shape(name)
            curshape = []
            if (index == 0 and engine.get_dynamic_input_type == aie.DynamicInputType.dynamic_dims):
                curshape = [1, 4, resolution[0]//8, resolution[1]//8]
            else:
                for i in range(curdims.size):
                    curshape.append(curdims[i])
            output_shapes.append(tuple(curshape))
        output_dtypes = [
            np_types[engine.get_data_type(name)] for name in output_names
        ]

        return RuntimeIOInfo(input_shapes, input_dtypes, output_shapes,
                             output_dtypes)

    @staticmethod
    def create_shared_buffers(shapes: List[tuple],
                              dtypes: List[type]) -> List[mp.RawArray]:
        buffers = []
        for shape, dtype in zip(shapes, dtypes):
            size = 1
            for x in shape:
                size *= x

            raw_array = mp.RawArray(np.ctypeslib.as_ctypes_type(dtype), size)
            buffers.append(raw_array)

        return buffers

    @staticmethod
    def run_infer(
        sync_pipe: mp.connection.Connection,
        input_spaces: List[np.ndarray],
        output_spaces: List[np.ndarray],
        io_info: RuntimeIOInfo,
        device_id: int,
        model_path: str,
        engine: aie.Engine,
    ) -> None:
        # The sub process function

        # Create a runtime
        ret = aie.set_device(device_id)
        runtime = aie.Runtime.get_instance()
        context = engine.create_context()
        buffer_binding = aie.IO_binding(engine, context)

        # Build numpy arrays on the shared buffers
        input_arrays = [
            np.frombuffer(b, dtype=t).reshape(s) for (b, s, t) in zip(
                input_spaces, io_info.input_shapes, io_info.input_dtypes)
        ]

        output_arrays = [
            np.frombuffer(b, dtype=t).reshape(s) for (b, s, t) in zip(
                output_spaces, io_info.output_shapes, io_info.output_dtypes)
        ]

        # Tell the main function that we are ready
        sync_pipe.send('')

        # Keep looping until recived a 'STOP'
        while sync_pipe.recv() != 'STOP':
            input_data_unet = [
                input_array for input_array in input_arrays
            ]
            output_data = aie.execute(input_data_unet, buffer_binding, context)

            for i, _ in enumerate(output_arrays):
                output_size = reduce(lambda x,y : x * y, io_info.output_shapes[i])
                output = np.frombuffer(output_data[i],
                                       dtype=io_info.output_dtypes[i])[:output_size].reshape(
                                           io_info.output_shapes[i])
                output_arrays[i][:] = output[i][:]

            sync_pipe.send('')
        aie.release_IO_binding(buffer_binding)
    
    @classmethod
    def clone(cls, device_id: int, model_path: str,
              engine: aie.Engine, resolution:List[int]) -> 'BackgroundRuntime':
        # Get shapes, datatypes from an existed engine,
        # then use them to create a BackgroundRuntime
        io_info = cls.get_io_info_from_engine(engine, resolution)
        if model_path[-2:] == "om":
            return cls(device_id, model_path, io_info, engine)
        else:
            return cls(device_id, model_path, io_info, engine) 

        
    def infer_asyn(self, feeds: List[np.ndarray]) -> None:
        for i, _ in enumerate(self.input_arrays):
            self.input_arrays[i][:] = feeds[i][:]

        self.sync_pipe.send('')

    def wait(self) -> None:
        self.sync_pipe.recv()

    def get_outputs(self) -> List[np.ndarray]:
        return self.output_arrays

    def wait_and_get_outputs(self) -> List[np.ndarray]:
        self.wait()
        return self.get_outputs()

    def infer(self, feeds: List[np.ndarray]) -> List[np.ndarray]:
        self.infer_asyn(feeds)
        return self.wait_and_get_outputs()

    def stop(self):
        # Stop the sub process
        self.sync_pipe.send('STOP')

