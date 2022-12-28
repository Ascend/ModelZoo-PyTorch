"""
Copyright 2020 Huawei Technologies Co., Ltd

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
import sys
import numpy as np
import pretrainedmodels
from pretrainedmodels import utils
from PIL import Image
from tqdm import tqdm


def init_transform():
    """build net transform"""
    model_name = 'nasnetalarge'
    net = pretrainedmodels.__dict__[model_name](
        num_classes=1001, pretrained='imagenet+background')
    net.eval()
    transform = utils.TransformImage(net)
    return transform


def preprecess(src_path, save_path):
    """
    :param src_path: input image root
    :param save_path: save bin dir
    """
    in_files = os.listdir(src_path)
    transform = init_transform()
    for out_file in tqdm(in_files):
        input_image = Image.open(src_path + '/' + out_file).convert('RGB')
        input_tensor = transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, out_file.split('.')[0] + ".bin"))


if __name__ == "__main__":
    input_path = sys.argv[1]
    out_path= sys.argv[2]
    if not os.path.isdir(out_path):
        os.makedirs(os.path.realpath(out_path))
    preprecess(input_path, out_path)
