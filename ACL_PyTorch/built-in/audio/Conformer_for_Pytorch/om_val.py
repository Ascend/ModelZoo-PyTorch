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

import argparse
from functools import partial
from multiprocessing import Pool, Manager
import numpy as np
from espnet_onnx import Speech2Text
from tqdm import tqdm
import os
import stat
import re
import time
import librosa


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                yield fullname


def dump_result(save_path, res_list,
                flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR):
    with os.fdopen(os.open(save_path, flags, mode), 'w') as f:
        for res in res_list:
            f.write(res)


def flat(datas):
    res = []
    for i in datas:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res


def shuffle_list(data_list):
    data_num = len(data_list[0])
    idxes = np.arange(data_num)
    np.random.shuffle(idxes)

    out_list = []
    for data in data_list:
        out_list.append(
            [data[idx] for idx in idxes]
        )
    return out_list


def load_data(data_dir, batch=1):
    data_list = []
    map_names = []
    for root, ds, fs in os.walk(data_dir):
        for f in fs:
            if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                data, sr = librosa.load(fullname, sr=16000)
                data_list.append(data)
                map_names.append(os.path.splitext(os.path.basename(fullname))[0])
    return data_list, map_names


def generate_batch_data(data_list, map_names, batch_num, device_ids, num_process,
                        **kwargs):
    # TODO: split data by num_process exactly
    # TODO: keep data more balance between mutli processes
    def pack_data(datas, names):
        if batch_num == 1:
            return datas, names
        packed_data = []
        packed_name = []
        data_num = len(datas)
        pad_impl = kwargs.get("pad_impl")
        for idx in range(0, data_num, batch_num):
            _d = datas[idx: idx+batch_num]
            _n = names[idx: idx+batch_num]
            if pad_impl is not None:
                _d = pad_impl(_d)
            packed_data.append(_d)
            packed_name.append(_n)
        return packed_data, packed_name

    data_list = flat(data_list)
    map_names = flat(map_names)
    if len(data_list) != len(map_names):
        raise ValueError("Num of map_names should be same as data_list.")

    sort = kwargs.get("sort")
    if sort:
        sort_impl = kwargs.get("sort_impl")
        if sort_impl is None:
            raise NotImplementedError("Please provide sort imply method.")
        data_list, map_names = sort_impl([data_list, map_names])

    # pack data by batch
    pad_impl = kwargs.get("pad_impl")
    packed_datas, packed_names = pack_data(data_list, map_names)
    if kwargs.get("shuffle"):
        packed_datas, packed_names = shuffle_list([packed_datas, packed_names])

    # split by process
    data_num = len(packed_datas)
    split_num = data_num // num_process + 1
    splited_packs = []
    num_per_device = data_num // len(device_ids) + 1
    for idx in range(0, data_num, split_num):
        device_id = device_ids[idx // num_per_device]
        if idx == num_process - 1:
            end_idx = data_num
        else:
            end_idx = idx + split_num
        splited_packs.append(
            (device_id, packed_datas[idx:end_idx], packed_names[idx:end_idx])
        )
    return splited_packs


def process_unsplited(args):
    # TODO: 支持多进程/存在部分精度差异
    speech2text = Speech2Text(
        model_dir=args.model_path,
        providers=["NPUExecutionProvider"],
        device_id=args.device_ids[0])
    total_t = 0
    files = findAllFile(args.dataset_path)
    files = list(files)
    num = len(files)
    res_list = []
    st = time.time()
    for fl in files:
        y, sr = librosa.load(fl, sr=16000)
        nbest = speech2text(y)
        res = "".join(nbest[0][1])
        res_list.append('{} {}\n'.format(fl.split('/')[-1].split('.')[0], res))
    et = time.time()
    dump_result(args.result_path, res_list)
    print("wav/second:", num/(et-st))


def infer_multi(data_pack, mode='default'):
    device_id, data_list, map_names = data_pack
    enable_multibatch = True if data_list[0].shape[0] > 1 else False
    only_use_decoder = False
    only_use_encoder = False
    if mode == 'decoder':
        only_use_decoder = True
        num_process = args.num_process_decoder
    elif mode == 'encoder':
        only_use_encoder = True
        num_process = args.num_process_encoder
    else:
        num_process = args.num_process

    speech2text = Speech2Text(
        model_dir=args.model_path,
        providers=["NPUExecutionProvider"],
        device_id=device_id,
        only_use_decoder=only_use_decoder,
        only_use_encoder=only_use_encoder
        enable_multibatch=enable_multibatch)

    sync_num.append(1)
    while (len(sync_num) != num_process):
        # keep sync between multi processes
        time.sleep(0.5)

    sample_num = 0
    res = []
    names_list = []
    st = time.time()
    for data_idx, datas in tqdm(enumerate(data_list)):
        names = map_names[data_idx]
        feature = speech2text(datas)
        if mode == 'encoder':
            feature = feature[0]
        elif mode == 'decoder':
            feature = [f[0] for f in feature]
        if isinstance(names, list):
            sample_num += len(names)
        else:
            sample_num += 1
        names_list.append(names)
        res.append(feature)
    et = time.time()
    sync_num.pop()
    return sample_num, (st, et), res, names_list


def process_splited(args):
    # encode process
    infer_encoder = partial(infer_multi, mode='encoder')
    sample_num = 0
    features = []
    encoder_times = []
    data_list, map_names = load_data(args.dataset_path)
    splited_packs = generate_batch_data(data_list, map_names, args.batch_encoder,
                                        args.device_ids, args.num_process_encoder,
                                        shuffle=True)
    args.num_process_encoder = min(len(splited_packs), args.num_process_encoder)
    map_names.clear()
    with Pool(args.num_process_encoder) as p:
        for _num, _duration, _fe, _name in list(p.imap(infer_encoder, splited_packs)):
            sample_num += _num
            map_names += _name
            encoder_times.append(_duration)
            features.append(_fe)

    # decode process
    def sort_by_feature(datas):
        features = datas[0]
        _res = []
        sorted_idxes = sorted(np.arange(len(features)), key=lambda x:features[x].shape[1])
        for data in datas:
            _res.append(
                [data[i] for i in sorted_idxes]
            )
        return _res

    def pad_by_zero(feature_list):
        max_len = max([_.shape[1] for _ in feature_list])
        out_list = [
            np.pad(_, ((0, 0), (0, max_len-_.shape[1]), (0,0))) \
            for _ in feature_list
        ]
        return np.concatenate(out_list, axis=0)

    infer_decoder = partial(infer_multi, mode='decoder')
    splited_packs = generate_batch_data(features, map_names, args.batch_decoder,
                                        args.device_ids, args.num_process_decoder, shuffle=True,
                                        sort=True, sort_impl=sort_by_feature, pad_impl=pad_by_zero)
    res = []
    name_list = []
    decoder_times = []
    args.num_process_decoder = min(len(splited_packs), args.num_process_decoder)
    with Pool(args.num_process_decoder) as p:
        for _, _duration, _re, _name in list(p.imap(infer_decoder, splited_packs)):
            decoder_times.append(_duration)
            name_list += _name
            res.extend(_re)

    res = flat(res)
    name_list = flat(name_list)
    out_list = [f"{name_list[i]} {res[i]}\n" for i in range(len(res))]
    encoder_duration = max(_[1] for _ in encoder_times) - min(_[0] for _ in encoder_times)
    decoder_duration = max(_[1] for _ in decoder_times) - min(_[0] for _ in decoder_times)
    total_fps = sample_num / (encoder_duration + decoder_duration)
    dump_result(args.result_path, out_list)
    print(f"sample_num:{sample_num}")
    print(f"encoder: {sample_num/encoder_duration}wav/second")
    print(f"decoder: {sample_num/decoder_duration}wav/second")
    print(f"total: {total_fps}wav/second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--dataset_path", default='test/S0768', type=str, help="datapath")
    parser.add_argument('--model_path', default="/root/.cache/espnet_onnx/asr_train_asr_qkv", type=str, help='path to the om model and config')
    parser.add_argument('--result_path', default="om.txt", type=str, help='path to result')
    parser.add_argument('--unsplit', action='store_false', help='enable unsplit mode')
    parser.add_argument('--num_process', default=1, type=int, help='num of process')
    parser.add_argument('--batch_encoder', default=1, type=int, help='num of encode process')
    parser.add_argument('--batch_decoder', default=16, type=int, help='num of encode process')
    parser.add_argument('--num_process_encoder', default=60, type=int, help='num of encode process')
    parser.add_argument('--num_process_decoder', default=14, type=int, help='num of decode process')
    parser.add_argument('--device_ids', default='0', type=str, help='device ids for NPU infer')
    parser.add_argument('--seed', default=123, type=int, help='seed for data shuffle')

    args = parser.parse_args()
    np.random.seed(args.seed)
    args.device_ids = [int(_) for _ in args.device_ids.split(",")]
    manager = Manager()
    sync_num = manager.list()

    if not args.unsplit:
        if args.batch_encoder > 1:
            raise NotImplementedError("Multibatch not supported now for encoder in splited mode.")
        process_splited()
    else:
        if args.num_process > 1 or len(args.device_ids) > 1:
            raise NotImplementedError("Multiprocess not supported now in unsplited mode.")
        process_unsplited()
