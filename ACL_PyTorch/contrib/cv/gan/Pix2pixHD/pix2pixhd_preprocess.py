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
from PIL import Image
import numpy as np
import multiprocessing
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)
    
def image_read(label_file, inst_file):                  
    A = Image.open(label_file) 
    A1 = A.load()
    transform_A = get_transform()
    A_tensor = transform_A(A) * 255.0
    inst = Image.open(inst_file)
    inst1 = inst.load()
    inst_tensor = transform_A(inst)
    
    return A_tensor, inst_tensor


def encode_input(label_map, inst_map, label_nc = 35, infer=True): 
    
    label_map = label_map.unsqueeze(0)
    inst_map = inst_map.unsqueeze(0)
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3]) 
    input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long(), 1.0)

    inst_map = inst_map.data
    edge_map = get_edges(inst_map)
    input_label = torch.cat((input_label, edge_map), dim=1)         
    input_label = Variable(input_label, volatile=infer)

    return input_label

def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])

    return edge.float()


def transform_invert(input_label):
    input_label = torch.squeeze(input_label, dim=0)
    input_label = input_label.detach().numpy() 
    float_input = input_label.astype(np.float32)

    return float_input


def gen_input_bin(label_batches, inst_batches, batch):
    count = 0
    for i in range(len(label_batches[batch])):
        count = count + 1
        print("i = ", i)
        print("len(label_batches[batch]) = ", len(label_batches[batch]))
        print("count = ", count)
        label_file = label_batches[batch][i]
        inst_file = inst_batches[batch][i]
        print("batch", batch, label_file, inst_file, "===", count)
        A_tensor, inst_tensor = image_read(label_file, inst_file)
        input_label = encode_input(A_tensor, inst_tensor)
        float_input = transform_invert(input_label)
        print("float_input = ",float_input.shape, float_input.dtype)
        print("save_path = ", os.path.join(save_path, label_file.split('.')[0] + ".bin"))
        float_input.tofile(os.path.join(save_path, label_file.split('.')[0] + ".bin"))



def preprocess(src_path, save_path):

    dir_label = 'test_label'  
    dir_label = os.path.join(src_path, dir_label)  
    label_paths = sorted(make_dataset(dir_label))  

    dir_inst = 'test_inst'
    dir_inst = os.path.join(src_path, dir_inst) 
    inst_paths = sorted(make_dataset(dir_inst))  

    for i in range(len(label_paths)):
        label_file = label_paths[i]
        inst_file = inst_paths[i]
        A_tensor, inst_tensor = image_read(label_file, inst_file)
        input_label = encode_input(A_tensor, inst_tensor)
        float_input = transform_invert(input_label)
        float_input.tofile(os.path.join(save_path, label_file.split('/')[-1].split('.')[0] + ".bin"))



if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("usage: python3 xxx.py [model_type] [src_path] [save_path]") 
    src_path = sys.argv[1]   
    save_path = sys.argv[2]  
    
    src_path = os.path.realpath(src_path)
    save_path = os.path.realpath(save_path)

    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))

    preprocess(src_path, save_path)
