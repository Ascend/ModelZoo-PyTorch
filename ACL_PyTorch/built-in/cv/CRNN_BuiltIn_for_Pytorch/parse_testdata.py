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

import os
import stat
import sys
import re
import six
import lmdb
from PIL import Image
import numpy as np
import torchvision


alphabets = '0123456789abcdefghijklmnopqrstuvwxyz'


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


def gen_data_label(test_dir, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    env = lmdb.open(test_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot open lmdb from %s' % (test_dir))
        sys.exit(0)
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()).decode('utf-8'))
        print('origin nSamples is:', nSamples)
        filtered_index_list = []

        with os.fdopen(os.open('label.txt', os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), 'w') as f:
            for index in range(nSamples):
                index += 1
                # images
                img_key = 'image-%09d'.encode() % index
                imgbuf = txn.get(img_key)
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf).convert('L')
                    img.show()
                    # transform
                    transform = resizeNormalize((100, 32))
                    img = transform(img)
                    img = np.array(img, np.float32)
                    img.tofile('{}/test_{}.bin'.format(data_dir, index))

                except IOError:
                    print('Corrupted image for %d' % index)

                # label
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')
                label = label.lower()

                line = 'test_{}.bin:{}'.format(index, label)
                f.write(line)
                f.write('\n')
                out_of_char = f'[^{alphabets}]'
                if re.search(out_of_char, label.lower()):
                    continue
                filtered_index_list.append(index)
        new_Samples = len(filtered_index_list)
        print('new nSamples is:', new_Samples)


if __name__ == '__main__':
    gen_data_label(sys.argv[1], sys.argv[2])
