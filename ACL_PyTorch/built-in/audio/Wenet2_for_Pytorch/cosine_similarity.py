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
import argparse

from pyacl.acl_infer import AclNet, init_acl, release_acl
import numpy as np
import onnxruntime

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b)/(a_norm * b_norm)
    return cos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get cosine similaryty with your model')
    parser.add_argument('--encoder_onnx', required=True, help='encoder onnx file')
    parser.add_argument('--encoder_om', required=True, help='encoder om file')
    parser.add_argument('--batch_size', required=True, type=int, help='batch size')
    parser.add_argument('--device_id', default=0, type=int, help='device id')
    parser.add_argument('--decoding_chunk_size', default=16, type=int, 
                        help='decoding chunk size, <=0 is not supported')
    parser.add_argument('--num_decoding_left_chunks',
                        default=4,
                        type=int,
                        required=False,
                        help="number of left chunks, <= 0 is not supported")
    args = parser.parse_args()
    print(args)

    init_acl(args.device_id)
    required_cache_size = args.decoding_chunk_size * args.num_decoding_left_chunks
    chunk_xs = np.random.random((args.batch_size, 67, 80)).astype("float32")
    chunk_lens = np.array([600]*args.batch_size).astype("int32")
    offset = np.array([0]*args.batch_size).reshape((args.batch_size, 1))
    att_cache = np.random.random((args.batch_size, 12, 4, required_cache_size, 128)).astype("float32")
    cnn_cache = np.random.random((args.batch_size, 12, 256, 7)).astype("float32")
    cache_mask = np.random.random((args.batch_size, 1, required_cache_size)).astype("float32")
    net = AclNet(model_path=args.encoder_om, device_id=args.device_id)
    output_data, exe_time = net([chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask])
    y = output_data[0].flatten()
    onnx_model = onnxruntime.InferenceSession(args.encoder_onnx)
    inputs = {
        onnx_model.get_inputs()[0].name: chunk_xs,
        onnx_model.get_inputs()[1].name: chunk_lens,
        onnx_model.get_inputs()[2].name: offset,
        onnx_model.get_inputs()[3].name: att_cache,
        onnx_model.get_inputs()[4].name: cnn_cache,
        onnx_model.get_inputs()[5].name: cache_mask,

    }
    onnx_output = onnx_model.run(None, inputs)
    onnx_y = onnx_output[0].flatten()
    cos_1 = cos_sim(y, onnx_y)
    print("acc: ", cos_1)
    net.release_model()
    release_acl(args.device_id)
