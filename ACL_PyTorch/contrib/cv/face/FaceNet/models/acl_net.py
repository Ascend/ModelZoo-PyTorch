# Copyright 2021 Huawei Technologies Co., Ltd
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

import numpy as np
import acl

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

# format
ACL_FORMAT_NCHW = 0
ACL_FLOAT32 = 0
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


def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret = {}".format(message, ret))


class Net(object):
    def __init__(self, context, device_id, model_path, input_dtype=ACL_FLOAT32, output_dtype=ACL_FLOAT32):
        self.device_id = device_id
        self.model_path = model_path
        self.model_id = None
        self.context = context
        self.buffer_method = {
            "in": acl.mdl.get_input_size_by_index,
            "out": acl.mdl.get_output_size_by_index,
            "outhost": acl.mdl.get_output_size_by_index
        }

        self.input_data = []
        self.output_data = []
        self.output_data_host = []
        self.model_desc = None
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.input_size = None
        self.output_size = None

        self.input_dtype = ACL_FLOAT32
        self.output_dtype = ACL_FLOAT32

        self._init_resource()

    def __call__(self, ori_data, out_size):
        return self.forward(ori_data, out_size)

    def __del__(self):
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

    def _release_data_buffer(self):
        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data_host:
            item = self.output_data_host.pop()
            ret = acl.rt.free_host(item["buffer"])
            check_ret("acl.rt.free_host", ret)

    def _init_resource(self):
        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()

    def _get_model_info(self):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        self.input_size = acl.mdl.get_num_inputs(self.model_desc)
        self.output_size = acl.mdl.get_num_outputs(self.model_desc)

    def _gen_data_buffer(self, size, des, data=None, out_size_list=None):
        func = self.buffer_method[des]
        for i in range(size):
            if out_size_list is None and data is None:
                temp_buffer_size = func(self.model_desc, i)
            else:
                if des == "in":
                    input_size = np.prod(np.array(data).shape)
                    temp_buffer_size = Net.gen_data_size(input_size, dtype=ACL_DTYPE.get(self.input_dtype))
                elif des == "out":
                    out_size = out_size_list[i]
                    temp_buffer_size = Net.gen_data_size(out_size, dtype=ACL_DTYPE.get(self.output_dtype))

            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _gen_dataset_output_host(self, size, des, out_size_list=None):
        func = self.buffer_method[des]
        for i in range(size):
            if out_size_list is None:
                temp_buffer_size = func(self.model_desc, i)
            else:
                out_size = out_size_list[i]
                temp_buffer_size = Net.gen_data_size(out_size, dtype=ACL_DTYPE.get(self.output_dtype))
                temp_buffer, ret = acl.rt.malloc_host(temp_buffer_size)
                check_ret("acl.rt.malloc_host", ret)

            self.output_data_host.append({"buffer": temp_buffer,
                                          "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data

        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            dataset = self.output_data_host

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                if 'bytes_to_ptr' in dir(acl.util):
                    bytes_in = dataset[i].tobytes()
                    ptr = acl.util.bytes_to_ptr(bytes_in)
                else:
                    ptr, _ = acl.util.numpy_contiguous_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"], item["size"], ptr, item["size"], policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr, item["size"], item["buffer"], item["size"], policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input", input_shapes=None):
        dataset = acl.mdl.create_dataset()
        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for i, item in enumerate(temp_dataset):
            data = acl.create_data_buffer(item["buffer"], item["size"])
            if data is None:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

            _, ret = acl.mdl.add_dataset_buffer(dataset, data)
            if ret != ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(dataset)
                check_ret("acl.destroy_data_buffer", ret)

            if type_str == "in":
                # set dynamic dataset tensor desc
                input_shape = input_shapes[i]
                input_desc = acl.create_tensor_desc(self.input_dtype, input_shape, ACL_FORMAT_NCHW)
                dataset, ret = acl.mdl.set_dataset_tensor_desc(dataset, input_desc, i)
                if ret != ACL_ERROR_NONE:
                    ret = acl.destroy_data_buffer(dataset)
                    check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        input_shapes = [list(data.shape) for data in images]
        self._gen_dataset("in", input_shapes)
        self._gen_dataset("out")

    def _data_from_device_to_host(self, input_data, out_size_list):
        res = []
        self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        output = self.get_result(self.output_data_host, input_data, out_size_list)
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

    def _prepare_data_buffer(self, input_data=None, out_size_list=None):
        self._gen_data_buffer(self.input_size, des="in", data=input_data)
        self._gen_data_buffer(self.output_size, des="out", out_size_list=out_size_list)
        self._gen_dataset_output_host(self.output_size, des="outhost", out_size_list=out_size_list)

    def forward(self, input_data, out_size_list):
        if not isinstance(input_data, (list, tuple)):
            input_data = [input_data]

        self._prepare_data_buffer(input_data=input_data, out_size_list=out_size_list)
        self._data_from_host_to_device(input_data)
        ret = acl.mdl.execute(self.model_id, self.load_input_dataset, self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()
        result = self._data_from_device_to_host(input_data=input_data, out_size_list=out_size_list)
        self._release_data_buffer()
        return result

    def get_result(self, output_data, data, out_size_list):
        dataset = []
        batch_size = data[0].shape[0]
        for i in range(len(output_data)):
            dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
            check_ret("acl.mdl.get_output_dims", ret)

            data_shape = dims.get("dims")
            # fix dynamic batch size
            # data_shape[0] = batch_size
            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            # data_len =  functools.reduce(lambda x, y: x * y, data_shape)
            data_len = out_size_list[i]
            ftype = np.dtype(ACL_DTYPE.get(data_type))

            size = output_data[i]["size"]
            ptr = output_data[i]["buffer"]
            if 'ptr_to_bytes' in dir(acl.util):
                data = acl.util.ptr_to_bytes(ptr, size)
                np_arr = np.frombuffer(data, dtype=ftype, count=data_len)
            else:
                data = acl.util.ptr_to_numpy(ptr, (size,), 1)
                np_arr = np.frombuffer(
                    bytearray(data[:data_len * ftype.itemsize]), dtype=ftype, count=data_len)
            np_arr = np_arr.reshape(data_shape)
            dataset.append(np_arr)
        return dataset

    @staticmethod
    def gen_data_size(size, dtype):
        dtype = np.dtype(dtype)
        return int(size * dtype.itemsize)

    @staticmethod
    def fix_static_shape(input_shape, idx, value):
        if not isinstance(input_shape, list):
            input_shape = list(input_shape)
            input_shape[idx] = value
        return input_shape
