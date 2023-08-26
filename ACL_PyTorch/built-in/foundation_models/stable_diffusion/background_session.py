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
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ais_bench.infer.interface import InferSession


@dataclass
class SessionIOInfo:
    input_shapes: List[tuple]
    input_dtypes: List[type]
    output_shapes: List[tuple]
    output_dtypes: List[type]


@dataclass
class BackgroundInferSessionOptions:
    device_id: int
    model_path: str
    io_info: SessionIOInfo
    acl_json_path: Optional[str] = None
    debug: Optional[bool] = False
    loop: Optional[int] = 1


class BackgroundInferSession:
    def __init__(
        self, 
        device_id: int, 
        model_path: str, 
        io_info: SessionIOInfo,
        acl_json_path: Optional[str] = None, 
        debug: Optional[bool] = False, 
        loop: Optional[int] = 1,
    ):
        # Create a pipe for process synchronization
        self.sync_pipe, sync_pipe_peer = mp.Pipe(duplex=True)

        # Create shared buffers
        input_spaces = self.create_shared_buffers(io_info.input_shapes, io_info.input_dtypes)
        output_spaces = self.create_shared_buffers(io_info.output_shapes, io_info.output_dtypes)

        # Build numpy arrays on the shared buffers
        self.input_arrays = [np.frombuffer(b, dtype=t).reshape(s) for (
            b, s, t) in zip(input_spaces, io_info.input_shapes, io_info.input_dtypes)]
        self.output_arrays = [np.frombuffer(b, dtype=t).reshape(s) for (
            b, s, t) in zip(output_spaces, io_info.output_shapes, io_info.output_dtypes)]

        mp.set_start_method('spawn')
        self.p = mp.Process(
            target=self.run_session, 
            args=[sync_pipe_peer, input_spaces, output_spaces,
                  io_info, device_id, model_path,
                  acl_json_path, debug, loop]
        )
        self.p.start()

        # Wait until the sub process is ready
        self.wait()

    def infer_asyn(self, feeds: List[np.ndarray]) -> None:
        for i in range(len(self.input_arrays)):
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
        # This function should work as same as InferSession.infer()
        self.infer_asyn(feeds)
        return self.wait_and_get_outputs()

    def stop(self):
        # Stop the sub process
        self.sync_pipe.send('STOP')

    @classmethod
    def clone(cls, session: InferSession, device_id: int) -> 'BackgroundInferSession':
        # Get shapes, datatypes, and model path from an existed InferSession, 
        # then use them to create a BackgroundInferSession
        io_info = cls.get_io_info_from_session(session)

        return cls(device_id, session.model_path, io_info)

    @staticmethod
    def get_io_info_from_session(session: InferSession) -> SessionIOInfo:
        # Map aclruntime datatype to numpy datatype
        np_types = (np.float32, np.float16, np.int8, np.int32, 
                    np.uint8, '', np.int16, np.uint16, np.uint32, 
                    np.int64, np.uint64)

        # Get input shapes and datatypes
        inputs = session.get_inputs()
        input_shapes = [t.shape for t in inputs]
        input_dtypes = [np_types[t.datatype] for t in inputs]

        # Get output shapes and datatypes
        outputs = session.get_outputs()
        output_shapes = [t.shape for t in outputs]
        output_dtypes = [np_types[t.datatype] for t in outputs]

        return SessionIOInfo(input_shapes, input_dtypes, 
                             output_shapes, output_dtypes)

    @staticmethod
    def create_shared_buffers(shapes: List[tuple], dtypes: List[type]) -> List[mp.RawArray]:
        buffers = []
        for shape, dtype in zip(shapes, dtypes):
            size = 1
            for x in shape:
                size *= x

            raw_array = mp.RawArray(np.ctypeslib.as_ctypes_type(dtype), size)
            buffers.append(raw_array)

        return buffers

    @staticmethod
    def run_session(
        sync_pipe: mp.connection.Connection,
        input_spaces: List[np.ndarray],
        output_spaces: List[np.ndarray],
        io_info: SessionIOInfo,
        device_id: int, 
        model_path: str, 
        acl_json_path: Optional[str] = None, 
        debug: Optional[bool] = False, 
        loop: Optional[int] = 1,
    ) -> None:
        # The sub process function

        # Create an InferSession
        session = InferSession(
            device_id,
            model_path,
            acl_json_path,
            debug,
            loop
        )

        # Build numpy arrays on the shared buffers
        input_arrays = [np.frombuffer(b, dtype=t).reshape(s) for (
            b, s, t) in zip(input_spaces, io_info.input_shapes, io_info.input_dtypes)]

        output_arrays = [np.frombuffer(b, dtype=t).reshape(s) for (
            b, s, t) in zip(output_spaces, io_info.output_shapes, io_info.output_dtypes)]

        # Tell the main function that we are ready
        sync_pipe.send('')

        # Keep looping until recived a 'STOP'
        while sync_pipe.recv() != 'STOP':
            output = session.infer(input_arrays)
            for i in range(len(output_arrays)):
                output_arrays[i][:] = output[i][:]

            sync_pipe.send('')
