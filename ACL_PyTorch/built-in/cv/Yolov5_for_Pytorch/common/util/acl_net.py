# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import numpy as np
import acl
import functools
import time

# error code
ACL_ERROR_NONE = 0

# memory malloc code
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

# memory copy code
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

ACL_DTYPE = {
    0: 'float32',
    1: 'float16',
    2: 'int8',
    3: 'int32',
    4: 'uint8',
    6: 'int16',
    7: 'uint16',
    8: 'uint32',
    9: 'int64',
    10: 'uint64',
    11: 'float64',
    12: 'bool',
}

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
}


def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}".format(message, ret))


class Net(object):
    def __init__(self, model_path, device_id=0, config_path=None):
        self.device_id = device_id
        self.model_path = model_path
        self.model_id = None
        self.context = None

        self.input_data = []
        self.output_data = []
        self.model_desc = None
        self.load_input_dataset = None
        self.load_output_dataset = None

        self._init_resource(config_path)

    def __call__(self, ori_data):
        return self.forward(ori_data)

    def __del__(self):
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)

    def _init_resource(self, config_path=None):
        if config_path and os.path.exists(config_path):
            print("config path:", config_path)
            ret = acl.init(config_path)
        else:
            ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()

    def _get_model_info(self, ):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")

    def _gen_data_buffer(self, size, des):
        func = buffer_method[des]
        for i in range(size):
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data

        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc host, ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                if 'bytes_to_ptr' in dir(acl.util):
                    bytes_in = dataset[i].tobytes()
                    ptr = acl.util.bytes_to_ptr(bytes_in)
                else:
                    ptr = acl.util.numpy_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"], item["size"], ptr, item["size"], policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr, item["size"], item["buffer"], item["size"], policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            if data is None:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

            _, ret = acl.mdl.add_dataset_buffer(dataset, data)
            if ret != ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        self._gen_dataset("in")
        self._gen_dataset("out")

    def _data_from_device_to_host(self):
        res = []
        self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        output = self.get_result(res)
        return output

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue

            num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(num):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    def forward(self, input_data):
        if not isinstance(input_data, (list, tuple)):
            input_data = [input_data]

        self._data_from_host_to_device(input_data)
        st = time.time()
        ret = acl.mdl.execute(self.model_id, self.load_input_dataset, self.load_output_dataset)
        exe_t = time.time() - st
        check_ret("acl.mdl.execute", ret)

        self._destroy_databuffer()
        result = self._data_from_device_to_host()
        return result, exe_t

    def get_result(self, output_data):
        dataset = []
        for i in range(len(output_data)):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
            check_ret("acl.mdl.get_cur_output_dims", ret)

            data_shape = dims.get("dims")
            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            data_len = functools.reduce(lambda x, y: x * y, data_shape)
            ftype = np.dtype(ACL_DTYPE.get(data_type))

            size = output_data[i]["size"]
            ptr = output_data[i]["buffer"]
            if 'ptr_to_bytes' in dir(acl.util):
                data = acl.util.ptr_to_bytes(ptr, size)
                np_array = np.frombuffer(data, dtype=ftype, count=data_len)
            else:
                data = acl.util.ptr_to_numpy(ptr, (size,), 1)
                np_array = np.frombuffer(bytearray(data[:data_len * ftype.itemsize]), dtype=ftype, count=data_len)
            np_array = np_array.reshape(data_shape)
            dataset.append(np_array)
        return dataset
