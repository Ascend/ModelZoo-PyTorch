# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
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

from abc import ABCMeta, abstractmethod
from typing import ByteString

from StreamManagerApi import (StreamManagerApi, MxDataInput, MxProtobufIn,
                              InProtobufVector)


class AscendPredictor(metaclass=ABCMeta):
    def __init__(self, pipeline_conf, stream_name):
        self.pipeline_conf = pipeline_conf
        self.stream_name = stream_name

    def __enter__(self):
        self.stream_manager_api = StreamManagerApi()
        ret = self.stream_manager_api.InitManager()
        if ret != 0:
            raise Exception(f"Failed to init Stream manager, ret={ret}")

        # create streams by pipeline config file
        with open(self.pipeline_conf, 'rb') as f:
            pipeline_str = f.read()
        ret = self.stream_manager_api.CreateMultipleStreams(pipeline_str)
        if ret != 0:
            raise Exception(f"Failed to create Stream, ret={ret}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # destroy streams
        self.stream_manager_api.DestroyAllStreams()

    def send_protobuf_data(self, stream_name, in_plugin_id, data: ByteString):
        self.stream_manager_api.SendProtobuf(stream_name, in_plugin_id, data)

    def create_protobuf_data(self, key: ByteString, ptype: ByteString, data):
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = ptype
        protobuf.protobuf = data.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)

        return protobuf_vec

    def send_data(self, stream_name, in_plugin_id, data_input):
        unique_id = self.stream_manager_api.SendData(stream_name, in_plugin_id,
                                                     data_input)
        if unique_id < 0:
            raise Exception("Failed to send data to stream")
        return unique_id

    def create_data(self, data: ByteString):
        data_input = MxDataInput()
        data_input.data = data

        return data_input

    def get_result(self, stream_name, unique_id):
        result = self.stream_manager_api.GetResult(stream_name, unique_id)
        if result.errorCode != 0:
            raise Exception(
                f"GetResultWithUniqueId error."
                f"errorCode={result.errorCode}, msg={result.data.decode()}")
        return result

    def predict(self, data_item):
        print('>' * 30)
        pred = self._predict(data_item)
        print('<' * 30)

        return self.post_process(pred)

    @abstractmethod
    def _predict(self, data):
        pass

    @abstractmethod
    def post_process(self, pred):
        pass
