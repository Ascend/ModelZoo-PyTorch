# -*- coding: utf-8 -*-
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
import time
import os
import torch
import lmdb
import six
import sys
import re
import random
import torchvision
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import apex


def seed_everything():
    random.seed(1234)
    os.environ['PYTHONHASHSEED'] = str(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def get_optimizer(config, model):
    optimizer = None
    if config.TRAIN.OPTIMIZER == "adadelta":
        # optimizer = torch.optim.Adadelta(
        optimizer = apex.optimizers.NpuFusedAdadelta(
            filter(lambda p: p.requires_grad, model.parameters())
        )
    return optimizer


def create_output_folder(cfg):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    checkpoints_output_dir = root_output_dir / time_str / 'checkpoints'
    print('=> creating {}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = root_output_dir / time_str / 'log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    return (str(checkpoints_output_dir), str(tensorboard_log_dir))


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        for item in text:
            length.append(len(item))
            r = []
            for char in item:
                index = self.dict[char]
                # result.append(index)
                r.append(index)
            result.append(r)
        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)
        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)
        text = result_temp
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)
        transform = resizeNormalize((self.imgW, self.imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class lmdbDataset(Dataset):
    def __init__(self, config, is_train=True, transform=None):
        if is_train:
            self.root = config.DATASET.TRAIN_ROOT
        else:
            self.root = config.DATASET.TEST_ROOT
        self.alphabets = config.DATASET.ALPHABETS
        self.is_train = is_train
        self.env = lmdb.open(self.root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()).decode('utf-8'))
            self.nSamples = nSamples
            print('origin nSamples is:', nSamples)
            if not config.DATASET.DATA_FILTER:
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')
                    out_of_char = f'[^{self.alphabets}]'
                    if re.search(out_of_char, label.lower()):
                        continue
                    self.filtered_index_list.append(index)
                self.nSamples = len(self.filtered_index_list)
                print('new nSamples is:', self.nSamples)
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            if self.transform is not None:
                img = self.transform(img)
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            label = label.lower()
        return (img, label)
