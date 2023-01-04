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

import argparse
import os.path as osp
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, RandomHorizontalFlip, ToTensor, Compose
from PIL import Image

from reid import datasets


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)

    normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = Compose([
        RectScale(height, width),
        RandomHorizontalFlip(),
        ToTensor(),
        normalizer,
    ])

    test_transformer = Compose([
        RectScale(height, width),
        ToTensor(),
        normalizer,
    ])


    query_loader = DataLoader(
        Preprocessor(dataset.query, root=osp.join(dataset.images_dir, dataset.query_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir, dataset.gallery_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)


    return query_loader, gallery_loader


def data_preprocess(bin_filepath, dataloader):   
    if os.path.exists(bin_filepath) == False:
        os.mkdir(bin_filepath)
    else:
        print('dir exist!')

    count = 0 
    for i, (img, fname, pid, _) in enumerate(dataloader):
        for fn, pi in zip(fname, pid):
            fname_1 = bin_filepath + '/' + fn.split('.', 2)[0] + '.bin'
            img = np.array(img).astype(np.float32)
            img.tofile(fname_1)
        count = count + 1
    return count


def main(args):

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    query_loader, gallery_loader = \
        get_data(args.dataset, args.data_dir, args.height, 
                 args.width, args.batch_size, args.workers,
                 )

    count = data_preprocess('./gallery_preproc_data_Ascend310', gallery_loader)
    print('number of images(gallery):')
    print(count)

    count = data_preprocess('./query_preproc_data_Ascend310', query_loader)
    print('number of images(query):')
    print(count)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    main(parser.parse_args())
