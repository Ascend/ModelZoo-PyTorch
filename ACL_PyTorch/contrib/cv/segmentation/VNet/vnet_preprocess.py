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

import sys
import os
import shutil
import numpy as np
from torchvision import transforms
import SimpleITK as sitk
from tqdm import tqdm

MIN_BOUND = -1000
MAX_BOUND = 400

image_dict = {}
label_dict = {}

def truncate(image, min_bound, max_bound):
    image[image < min_bound] = min_bound
    image[image > max_bound] = max_bound
    return image

def get_subvolume(target, bounds):
    (zs, ze), (ys, ye), (xs, xe) = bounds
    return np.squeeze(target)[zs:ze, ys:ye, xs:xe]

def load_image(root, series):
    if series in image_dict.keys():
        return image_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    image_dict[series] = truncate(img, MIN_BOUND, MAX_BOUND)
    return img

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

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    test_list = open(test_list_path).read().split('\n')[:-1]

    normMu = [-642.794]
    normSigma = [459.512]
    normTransform = transforms.Normalize(normMu, normSigma)
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    image_dir = os.path.join(data_dir, 'normalized_lung_ct')
    target_dir = os.path.join(data_dir, 'normalized_lung_mask')
    infos, _ = make_dataset(target_dir, test_list, partition=[2,2,2], nonempty=True)

    for i in tqdm(range(len(infos))):
        series, bounds = infos[i]
        (zs, ze), (ys, ye), (xs, xe) = bounds
        image = load_image(image_dir, series)
        image = image[zs:ze, ys:ye, xs:xe]
        image = image.transpose([1,2,0])
        image = image.astype(np.float32)
        image = testTransform(image).numpy()
        image.tofile(os.path.join(output_dir, "%d.bin")%i)


if __name__ == "__main__":
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    test_list_path = sys.argv[3]
    main()
