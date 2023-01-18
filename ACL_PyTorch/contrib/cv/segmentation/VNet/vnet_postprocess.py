# Copyright 2020 Huawei Technologies Co., Ltd
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
import sys
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

label_dict = {}

def get_subvolume(target, bounds):
    (zs, ze), (ys, ye), (xs, xe) = bounds
    return np.squeeze(target)[zs:ze, ys:ye, xs:xe]

def load_label(root, series):
    if series in label_dict.keys():
        return label_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    if np.max(img) > 3400:
        img[img <= 3480] = 0
        img[img > 3480] = 1
    else:
        img[img != 0] = 1
    label_dict[series] = img.astype(np.uint8)
    return img

def make_dataset(target_dir, test_list, partition, nonempty):
    sample_label = load_label(target_dir, test_list[0])
    shape = np.shape(sample_label)
    part_list = []
    z, y, x = shape
    if partition is not None:
        z_p, y_p, x_p = partition
        z, y, x = shape
        z_incr, y_incr, x_incr = z // z_p, y // y_p, x // x_p
        assert z % z_p == 0
        assert y % y_p == 0
        assert x % x_p == 0
        for zi in range(z_p):
            zstart = zi*z_incr
            zend = zstart + z_incr
            for yi in range(y_p):
                ystart = yi*y_incr
                yend = ystart + y_incr
                for xi in range(x_p):
                    xstart = xi*x_incr
                    xend = xstart + x_incr
                    part_list.append(((zstart, zend), (ystart, yend), (xstart, xend)))
    else:
        part_list = [((0, z), (0, y), (0, x))]
    result = []
    target_means = []
    keys = test_list
    for key in keys:
        for part in part_list:
            target = load_label(target_dir, key)
            if nonempty:
                if np.sum(get_subvolume(target, part)) == 0:
                    continue
            target_means.append(np.mean(target))
            result.append((key, part))

    target_mean = np.mean(target_means)
    return (result, target_mean)

def main():
    test_list = open(test_list_path).read().split('\n')[:-1]
    infos, _ =make_dataset(target_dir, test_list, partition=[2,2,2], nonempty=True)
    incorrect = 0
    numel = 0
    for i in tqdm(range(len(infos))):
        result_path = os.path.join(result_dir, '%d_0.bin'%i)
        result = np.fromfile(result_path, dtype='float32')
        result = result.reshape(-1,2)
        result = result.argmax(1)
        series, bounds = infos[i]
        (zs, ze), (ys, ye), (xs, xe) = bounds
        target = load_label(target_dir, series)
        target = target[zs:ze, ys:ye, xs:xe]
        target = target.reshape(target.size)
        numel += target.size
        incorrect += (result!=target).sum()
    err = 100.*incorrect/numel
    print('Test set: Error: {}/{} ({:.4f}%)\n'.format(incorrect, numel, err))

if __name__ == "__main__":
    result_dir = sys.argv[1]
    target_dir = sys.argv[2]
    test_list_path = sys.argv[3]
    main()
