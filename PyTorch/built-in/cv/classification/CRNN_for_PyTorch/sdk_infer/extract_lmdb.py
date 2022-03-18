# coding: utf-8
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
import os
import argparse

import lmdb


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


def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path.rstrip('/')))[0]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", help="lmdb dataset dir")
    return parser.parse_args()


def main():
    args = parse_args()
    lmdb_dir = args.data_dir
    lmdb_name = get_file_name(lmdb_dir.rstrip('/'))
    lmdb_img_dir = f"{lmdb_name}_img"
    lmdb_label_fn = f"{lmdb_name}_gt.txt"
    os.makedirs(lmdb_img_dir, exist_ok=True)
    dataset = LmdbDataLoader(lmdb_dir, limit=None)
    with open(lmdb_label_fn, 'w') as lmdb_label_fd:
        for name, label, data in dataset:
            img = f'{lmdb_img_dir}/{name}.jpg'
            with open(img, 'wb') as fd:
                fd.write(data)
            lmdb_label_fd.write(f"{name} {label}\n")
    print(f"extrace {lmdb_name} to {lmdb_img_dir} success.")
    print(f"gt: {lmdb_label_fn}")
    print(f"Total: {dataset.n_samples}")


if __name__ == '__main__':
    main()
