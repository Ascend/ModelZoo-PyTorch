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

import functools
import time
import acl
import numpy as np
import torch

# error code
ACL_ERROR_NONE = 0

# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

# dtype
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
        raise Exception(f"{message} failed ret={ret}")


class MeasureTime():
    def __init__(self, measurements, key, cpu_run=True):
        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

    def __enter__(self):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter_ns() - self.t0


class AclModel(object):
    def __init__(self, device_id, model_path, sync_infer, measurements, key, cpu_run):
        self.device_id = device_id
        self.sync_infer = sync_infer
        self.out_bufs_ptr = []
        self.output_sizes = []
        self.input_sizes = []
        self.input_bufs_ptr = []

        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

        ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)
        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        assert self.model_desc is not None
        acl.mdl.get_desc(self.model_desc, self.model_id)
        self.dataset_in = acl.mdl.create_dataset()
        assert self.dataset_in is not None
        self.dataset_out = acl.mdl.create_dataset()
        assert self.dataset_out is not None
        self.in_size, self.out_size = 0, 0
        self.stm, ret = acl.rt.create_stream()
        assert ret == 0

        self.desc_init()
        self.dataset_init()

    def __call__(self, ori_data, dim):
        return self.forward(ori_data, dim)

    def __del__(self):
        # unload model
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            assert ret == 0

        # destroy model desc
        ret = acl.mdl.destroy_desc(self.model_desc)
        assert ret == 0

        self.destroy_data_set(self.dataset_in)
        self.destroy_data_set(self.dataset_out)

        # destroy input/output tensor
        for i in range(len(self.input_bufs_ptr)):
            acl.rt.free(self.input_bufs_ptr[i]["buffer"])
            self.input_bufs_ptr[i] = None

        for i in range(len(self.out_bufs_ptr)):
            acl.rt.free(self.out_bufs_ptr[i]["buffer"])
            self.out_bufs_ptr[i] = None

        ret = acl.rt.destroy_stream(self.stm)
        assert ret == 0

    def desc_init(self):
        tensor_size = acl.mdl.get_num_inputs(self.model_desc)
        if not tensor_size:
            raise Exception("get_num_inputs failed")
        self.in_size = tensor_size

        for i in range(tensor_size):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            data, ret = acl.rt.malloc(size, 0)
            assert ret == 0

            self.input_bufs_ptr.append({'size': size, 'buffer': data})
            self.input_sizes.append(size)

        tensor_size = acl.mdl.get_num_outputs(self.model_desc)
        self.out_size = tensor_size
        for i in range(tensor_size):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
            assert ret == 0
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)

            data, ret = acl.rt.malloc(size, 0)
            assert ret == 0

            self.output_sizes.append(size)
            self.out_bufs_ptr.append({'size': size, 'buffer': data})

    def dataset_init(self):
        self.create_data_set(self.dataset_in, self.input_bufs_ptr, self.input_sizes)
        self.create_data_set(self.dataset_out, self.out_bufs_ptr, self.output_sizes)

    def create_data_set(self, dataset, bufs_ptr_list, size_list):
        # create dataset buffer then add to dataset
        for i, x in enumerate(size_list):
            buffer = acl.create_data_buffer(bufs_ptr_list[i]["buffer"], x)
            if not buffer:
                self.destroy_data_set(dataset)
                raise Exception("create_data_buffer failed")

            # add to dataset
            _, ret = acl.mdl.add_dataset_buffer(dataset, buffer)
            if ret != 0:
                self.destroy_data_set(dataset)
                raise Exception("add_dataset_buffer failed, ret = {}".format(ret))

        return dataset

    def destroy_data_set(self, dataset):
        data_buf_num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(data_buf_num):
            # get data buffer by index
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buf is not None:
                acl.destroy_data_buffer(data_buf)

        acl.mdl.destroy_dataset(dataset)

    def copy_data_to_device(self, data):
        for i, x in enumerate(data):
            if 'bytes_to_ptr' in dir(acl.util):
                data_in = x["buffer"].tobytes()
                ptr = acl.util.bytes_to_ptr(data_in)
            else:
                ptr, tmp = acl.util.numpy_contiguous_to_ptr(x["buffer"])
            acl.rt.memcpy(self.input_bufs_ptr[i]["buffer"], x["size"], ptr,
                          x["size"], ACL_MEMCPY_HOST_TO_DEVICE)

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

    def model_exe(self):
        with MeasureTime(self.measurements, self.key, self.cpu_run):
            ret = acl.mdl.execute(self.model_id, self.dataset_in, self.dataset_out)
        assert ret == 0
        output_data = self.copy_output_to_host()
        dataset = []
        for i, x in enumerate(output_data):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
            data_shape = dims.get("dims")
            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            data_len = functools.reduce(lambda x, y: x * y, data_shape)
            ftype = np.dtype(ACL_DTYPE.get(data_type))

            size = x["size"]
            ptr = x["buffer"]
            if 'ptr_to_bytes' in dir(acl.util):
                data = acl.util.ptr_to_bytes(ptr, size)
                np_arr = np.frombuffer(data, dtype=ftype, count=data_len)
            else:
                data = acl.util.ptr_to_numpy(ptr, (size,), 1)
                np_arr = np.frombuffer(bytearray(data[:data_len * ftype.itemsize]), dtype=ftype, count=data_len)
            np_arr = np_arr.reshape(data_shape)
            dataset.append(np_arr)
        return dataset

    def model_exe_async(self):
        with MeasureTime(self.measurements, self.key, self.cpu_run):
            ret = acl.mdl.execute_async(self.model_id, self.dataset_in, self.dataset_out, self.stm)
        assert ret == 0
        ret = acl.rt.synchronize_stream(self.stm)
        assert ret == 0
        output_data = self.copy_output_to_host()

        dataset = []
        for i, x in enumerate(output_data):
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
            data_shape = dims.get("dims")

            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            data_len = functools.reduce(lambda x, y: x * y, data_shape)
            ftype = np.dtype(ACL_DTYPE.get(data_type))

            size = x["size"]
            ptr = x["buffer"]
            if 'ptr_to_bytes' in dir(acl.util):
                data = acl.util.ptr_to_bytes(ptr, size)
                np_arr = np.frombuffer(data, dtype=ftype, count=data_len)
            else:
                data = acl.util.ptr_to_numpy(ptr, (size,), 1)
                np_arr = np.frombuffer(bytearray(data[:data_len * ftype.itemsize]), dtype=ftype, count=data_len)
            np_arr = np_arr.reshape(data_shape)
            dataset.append(np_arr)
        return dataset

    def model_exe_with_dynamic_dims(self, input_data, dims):
        index, ret = acl.mdl.get_input_index_by_name(self.model_desc, 'ascend_mbatch_shape_data')
        ret = acl.mdl.set_input_dynamic_dims(self.model_id, self.dataset_in, index, dims)
        gear_count, ret = acl.mdl.get_input_dynamic_gear_count(self.model_desc, -1)
        dims_out, ret = acl.mdl.get_input_dynamic_dims(self.model_desc, -1, gear_count)
        self.copy_data_to_device(input_data)
        if self.sync_infer is True:
            res = self.model_exe()
        else:
            res = self.model_exe_async()

        return res

    def forward(self, input_data, dims):
        input_data_dic = []
        for i, x in enumerate(input_data):
            temp = {}
            temp["size"] = x.size * x.itemsize
            temp["buffer"] = x
            input_data_dic.append(temp)
        result = self.model_exe_with_dynamic_dims(input_data_dic, dims)
        return result
