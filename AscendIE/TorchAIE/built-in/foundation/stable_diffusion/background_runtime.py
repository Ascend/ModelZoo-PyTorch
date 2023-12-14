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

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch_aie


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
            io_info: RuntimeIOInfo
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
                                io_info, device_id, model_path
                            ])
        self.p.start()

        # Wait until the sub process is ready
        self.wait()

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

    def infer_asyn(self, feeds: List[np.ndarray]) -> None:
        for i, _ in enumerate(self.input_arrays):
            print(f'bg input shape: {self.input_arrays[i].shape}')
            print(f'feeds shape: {feeds[i].shape}')
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

    @staticmethod
    def run_infer(
            sync_pipe: mp.connection.Connection,
            input_spaces: List[np.ndarray],
            output_spaces: List[np.ndarray],
            io_info: RuntimeIOInfo,
            device_id: int,
            model_path: str,
    ) -> None:
        # The sub process function

        # Create a runtime
        # Create a runtime
        torch_aie.set_device(device_id)
        print(f"[info] bg device id: {device_id}")

        # Tell the main function that we are ready
        model = torch.jit.load(model_path).eval()

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

        infer_num = 0
        preprocess_time = 0
        infer_time = 0
        forward_time = 0

        stream = torch_aie.npu.Stream(f"npu:{device_id}")

        # Keep looping until recived a 'STOP'
        while sync_pipe.recv() != 'STOP':
            start = time.time()
            sample, timestep, encoder_hidden_states = [
                torch.Tensor(input_array) for input_array in input_arrays
            ]
            sample_npu = sample.to(torch.float32).to(f"npu:{device_id}")
            timestep_npu = timestep.to(torch.int64).to(f"npu:{device_id}")
            encoder_hidden_states_npu = encoder_hidden_states.to(torch.float32).to(f"npu:{device_id}")
            preprocess_time += time.time() - start

            start2 = time.time()
            with torch_aie.npu.stream(stream):
                inf_start = time.time()
                output_npu = model(sample_npu, timestep_npu, encoder_hidden_states_npu)
                stream.synchronize()
                inf_end = time.time()

            output_cpu = output_npu.to('cpu')
            forward_time += inf_end - inf_start
            infer_time += time.time() - start2

            for i, _ in enumerate(output_arrays):
                output = output_cpu.numpy()
                output_arrays[i][:] = output[i][:]

            infer_num += 1
            sync_pipe.send('')

        infer_num /= 50
        print(f""
              f"bg preprocess_time time: {preprocess_time / infer_num:.3f}s\n"
              f"bg forward time: {forward_time / infer_num:.3f}s\n"
              f"bg infer time: {infer_time / infer_num:.3f}s\n"
              )

    @classmethod
    def clone(cls, device_id: int, model_path: str, runtime_info: RuntimeIOInfo) -> 'BackgroundRuntime':
        # Get shapes, datatypes from an existed engine,
        # then use them to create a BackgroundRuntime
        return cls(device_id, model_path, runtime_info)
