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
import functools

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
NPY_BYTE = 1
buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
}

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


def check_ptr(message, ptr):
    if ptr is None:
        raise Exception("{} failed, ptr is None".format(message))


class NetDynamic(object):
    def __init__(self, device_id, model_path):
        self.device_id = device_id
        self.out_bufs_ptr = []
        self.output_sizes = []
        self.input_sizes = []
        self.input_bufs_ptr = []

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        check_ptr("acl.mdl.create_desc", self.model_desc)
        acl.mdl.get_desc(self.model_desc, self.model_id)
        self.dataset_in = acl.mdl.create_dataset()
        check_ptr("acl.mdl.create_dataset", self.dataset_in)
        self.dataset_out = acl.mdl.create_dataset()
        check_ptr("acl.mdl.create_dataset", self.dataset_out)
        self.in_size, self.out_size = 0, 0
        self.stm, ret = acl.rt.create_stream()
        check_ret("acl.mdl.create_stream", ret)

        self.desc_init()
        self.dataset_init()

    def __call__(self, ori_data, dim):
        return self.forward(ori_data, dim)

    def __del__(self):
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)

        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)
            self.model_desc = None

        self.destroy_data_set(self.dataset_in)
        self.destroy_data_set(self.dataset_out)

        for i in range(len(self.input_bufs_ptr)):
            acl.rt.free(self.input_bufs_ptr[i]["buffer"])
            self.input_bufs_ptr[i] = None

        for i in range(len(self.out_bufs_ptr)):
            acl.rt.free(self.out_bufs_ptr[i]["buffer"])
            self.out_bufs_ptr[i] = None

        ret = acl.rt.destroy_stream(self.stm)
        check_ret("acl.rt.destroy_stream", ret)

    def desc_init(self):
        tensor_size = acl.mdl.get_num_inputs(self.model_desc)
        if not tensor_size:
            raise Exception("get_num_inputs failed")
        self.in_size = tensor_size

        for i in range(tensor_size):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            data, ret = acl.rt.malloc(size, 0)
            check_ret("acl.rt.malloc", ret)

            self.input_bufs_ptr.append({'size': size, 'buffer': data})
            self.input_sizes.append(size)

        tensor_size = acl.mdl.get_num_outputs(self.model_desc)
        self.out_size = tensor_size
        for i in range(tensor_size):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
            check_ret("acl.mdl.get_cur_output_dims", ret)
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            data, ret = acl.rt.malloc(size, 0)
            check_ret("acl.rt.malloc", ret)

            self.output_sizes.append(size)
            self.out_bufs_ptr.append({'size': size, 'buffer': data})

    def dataset_init(self):
        self.create_data_set(self.dataset_in, self.input_bufs_ptr, self.input_sizes)
        self.create_data_set(self.dataset_out, self.out_bufs_ptr, self.output_sizes)

    def create_data_set(self, dataset, bufs_ptr_list, size_list):
        for i in range(len(size_list)):
            buffer = acl.create_data_buffer(bufs_ptr_list[i]["buffer"], size_list[i])
            if not buffer:
                self.destroy_data_set(dataset)
                raise Exception("create_data_buffer failed")

            _, ret = acl.mdl.add_dataset_buffer(dataset, buffer)
            if ret != ACL_ERROR_NONE:
                self.destroy_data_set(dataset)
                raise Exception("add_dataset_buffer failed, ret = {}".format(ret))

        return dataset

    def destroy_data_set(self, dataset):
        data_buf_num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(data_buf_num):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buf is not None:
                acl.destroy_data_buffer(data_buf)

        acl.mdl.destroy_dataset(dataset)

    def copy_data_to_device(self, data):
        for i in range(len(data)):
            if 'bytes_to_ptr' in dir(acl.util):
                data_in = data[i]["buffer"].tobytes()
                ptr = acl.util.bytes_to_ptr(data_in)
            else:
                ptr, np = acl.util.numpy_contiguous_to_ptr(data[i]["buffer"])
            acl.rt.memcpy(self.input_bufs_ptr[i]["buffer"], data[i]["size"], ptr,
                          data[i]["size"], ACL_MEMCPY_HOST_TO_DEVICE)

    def copy_output_to_host(self):
        output_data = []
        for i in range(len(self.out_bufs_ptr)):
            temp = dict()
            temp["size"] = self.out_bufs_ptr[i]["size"]
            temp["buffer"], ret = acl.rt.malloc_host(temp["size"])
            output_data.append(temp)
            acl.rt.memcpy(temp["buffer"], temp["size"], self.out_bufs_ptr[i]["buffer"],
                          temp["size"], ACL_MEMCPY_DEVICE_TO_HOST)
        return output_data

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
                np_arr = np.frombuffer(data, dtype=ftype, count=data_len)
            else:
                data = acl.util.ptr_to_numpy(ptr, (size,), 1)
                np_arr = np.frombuffer(bytearray(data[:data_len * ftype.itemsize]), dtype=ftype, count=data_len)
            np_arr = np_arr.reshape(data_shape)
            dataset.append(np_arr)
        return dataset

    def model_exe_asyn(self):
        ret = acl.mdl.execute_async(self.model_id, self.dataset_in, self.dataset_out, self.stm)
        check_ret("acl.mdl.execute", ret)
        ret = acl.rt.synchronize_stream(self.stm)
        check_ret("acl.rt.synchronize_stream", ret)
        output_data = self.copy_output_to_host()

        dataset = []
        for i in range(len(output_data)):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
            data_shape = dims.get("dims")

            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            data_len = functools.reduce(lambda x, y: x * y, data_shape)
            ftype = np.dtype(ACL_DTYPE.get(data_type))

            size = output_data[i]["size"]
            ptr = output_data[i]["buffer"]
            if 'ptr_to_bytes' in dir(acl.util):
                data = acl.util.ptr_to_bytes(ptr, size)
                np_arr = np.frombuffer(data, dtype=ftype, count=data_len)
            else:
                data = acl.util.ptr_to_numpy(ptr, (size,), 1)
                np_arr = np.frombuffer(bytearray(data[:data_len * ftype.itemsize]), dtype=ftype, count=data_len)
            np_arr = np_arr.reshape(data_shape)
            dataset.append(np_arr)
        return dataset

    def copy_data_to_device_dynamic_batch(self, image_ptr_list, image_size, batch, max_batch):
        single_buffer_size = self.input_sizes[0] // max_batch
        stride = 0
        for i in range(batch):
            ret = acl.rt.memcpy(self.input_bufs_ptr[0]["buffer"] + stride, image_size,
                                image_ptr_list[i], image_size, ACL_MEMCPY_HOST_TO_DEVICE)
            check_ret("acl.rt.memcpy", ret)
            stride += single_buffer_size

    def model_exe_with_dynamic_dims(self, input_data, dims):
        index, ret = acl.mdl.get_input_index_by_name(self.model_desc, 'ascend_mbatch_shape_data')
        ret = acl.mdl.set_input_dynamic_dims(self.model_id, self.dataset_in, index, dims)
        gear_count, ret = acl.mdl.get_input_dynamic_gear_count(self.model_desc, -1)
        dims_out, ret = acl.mdl.get_input_dynamic_dims(self.model_desc, -1, gear_count)
        self.copy_data_to_device(input_data)
        return self.model_exe_asyn()

    def forward(self, input_data, dims):
        input_data_dict = []
        for i in range(len(input_data)):
            temp = {}
            temp["size"] = input_data[i].size * input_data[i].itemsize
            temp["buffer"] = input_data[i]
            input_data_dict.append(temp)

        return self.model_exe_with_dynamic_dims(input_data_dict, dims)