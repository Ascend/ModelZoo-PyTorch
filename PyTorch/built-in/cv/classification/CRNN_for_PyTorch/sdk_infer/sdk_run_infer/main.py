#/usr/bin/env python
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import json
import os
from contextlib import ExitStack

import lmdb
from StreamManagerApi import *


class LmdbDataLoader(object):
    def __init__(self, data_dir, limit=None, transforms=None):
        self.limit = limit
        self.data_dir = data_dir
        self.alphaberts = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.data_filter = True
        self.env = lmdb.open(data_dir, readonly=True)
        self.transforms = transforms

        self.n_samples = self.get_n_samples(limit)
        self.cur_index = 1

    def get_n_samples(self, limit):
        with self.env.begin(write=False) as txn:
            n_samples = int(txn.get(b'num-samples').decode('utf-8'))
        return n_samples if self.limit is None else min(n_samples, limit)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_index > self.n_samples:
            raise StopIteration()

        index = self.cur_index
        self.cur_index += 1
        with self.env.begin(write=False) as txn:
            img_key = b'image-%09d' % index
            imgbuf = txn.get(img_key)
            label_key = b'label-%09d' % index
            label = txn.get(label_key).decode('utf-8').lower()

        print(f"read img {img_key} {label}")
        return img_key.decode('utf-8'), label, imgbuf


class Predictor(object):
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
        self.data_input = MxDataInput()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # destroy streams
        self.stream_manager_api.DestroyAllStreams()

    def predict(self, dataset):
        print("Start predict........")
        print('>' * 30)
        for name, lable, data in dataset:
            self.data_input.data = data
            yield self._predict(name, self.data_input)
        print("predict end.")
        print('<' * 30)

    def _predict(self, name, data):
        plugin_id = 0
        unique_id = self._predict_send_data(self.stream_name, plugin_id, data)
        result = self._predict_get_result(self.stream_name, unique_id)
        return name, json.loads(result.data.decode())

    def _predict_send_data(self, stream_name, in_plugin_id, data_input):
        unique_id = self.stream_manager_api.SendData(stream_name, in_plugin_id,
                                                     data_input)
        if unique_id < 0:
            raise Exception("Failed to send data to stream")
        return unique_id

    def _predict_get_result(self, stream_name, unique_id):
        result = self.stream_manager_api.GetResult(stream_name, unique_id)
        if result.errorCode != 0:
            raise Exception(
                f"GetResultWithUniqueId error."
                f"errorCode={result.errorCode}, msg={result.data.decode()}")
        return result


def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path.rstrip('/')))[0]


def result_encode(file_name, result):
    if "MxpiAttribute" in result:
        texts = result.get("MxpiAttribute", [{}])
        pred = ''.join(texts[0].get('attrValue', ''))
    elif "MxpiTextsInfo" in result:
        texts = result.get("MxpiTextsInfo", [{}])
        pred = ''.join(texts[0].get('text', ''))
    else:
        pred = '-'
    return f"{file_name} {pred}\n"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', help='prediction data dir')
    parser.add_argument('result_file', help='result file')
    return parser.parse_args()


def main():
    pipeline_conf = "crnn_opencv.pipeline.json"
    stream_name = b'classification'

    args = parse_args()
    result_fname = get_file_name(args.result_file)
    pred_result_file = f"{result_fname}.txt"
    dataset = LmdbDataLoader(args.data_dir, limit=None)
    with ExitStack() as stack:
        predictor = stack.enter_context(Predictor(pipeline_conf, stream_name))
        result_fd = stack.enter_context(open(pred_result_file, 'w'))

        for fname, pred_result in predictor.predict(dataset):
            result_fd.write(result_encode(fname, pred_result))

    print(f"success, result in {pred_result_file}")


if __name__ == "__main__":
    main()
