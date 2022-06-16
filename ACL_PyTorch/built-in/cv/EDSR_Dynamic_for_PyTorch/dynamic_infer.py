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
import argparse
import numpy as np
from tqdm import tqdm
from acl_infer import AclNet, init_acl, release_acl


parser = argparse.ArgumentParser(description='Dynamic infer script')
parser.add_argument('-i', '--input_dir', default='./data_preprocessed/B100/npy',
                    type=str, help='input dir for preprocessed data')
parser.add_argument('-m', '--model_path', default='./EDSR_x2.om',
                    type=str, help='high res path')
parser.add_argument('-s', '--save_dir', default='./outputs/B100',
                    type=str, help='save dir for infer results')
parser.add_argument('--device_id', default=0,
                    type=int, help='device id')
parser.add_argument('--max_out_size', default=1080000,
                    type=int, help='max out size for infer results, default: 1*3*300*300*4')
args = parser.parse_args()


def generate_data():
    input_list = []
    for input_name in sorted(os.listdir(input_dir)):
        if os.path.splitext(input_name)[1] == ".npy":
            input_path = os.path.join(input_dir, input_name)
            input_data = np.load(input_path)
            input_list.append((input_name, input_data))
    return input_list


def main():
    input_list = generate_data()
    for idx, pack in tqdm(enumerate(input_list)):
        data_name, input_data = pack
        input_data = np.expand_dims(input_data, 0)
        output_data, time = om_model.forward(input_data, input_data.shape)
        save_path = os.path.join(save_dir, "{}.npy".format(
            os.path.splitext(data_name)[0]))
        print("{} data, shape: {}, time: {}".format(idx, input_data.shape, time))
        np.save(save_path, output_data[0][0])


if __name__ == '__main__':
    input_dir = args.input_dir
    model_path = args.model_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    init_acl(args.device_id)
    om_model = AclNet(
        model_path=model_path, device_id=args.device_id, output_data_shape=args.max_out_size
    )
    main()
    release_acl(args.device_id)
