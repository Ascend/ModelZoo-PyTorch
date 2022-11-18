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

import os
import numpy as np
import torch
import argparse
import sys
sys.path.append('./models')
from tqdm import tqdm
from models.pointnet2_utils import sample_and_group, farthest_point_sample
import glob

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('data_process')
    parser.add_argument('--preprocess_part', type=int, default=1, required=False, help='preprocess target')
    parser.add_argument('--save_path', type=str, default='./modelnet40_processed/test_preprocess/pointset_chg', required=False, help='target root')
    parser.add_argument('--save_path2', type=str, default='./modelnet40_processed/test_preprocess/xyz_chg', required=False, help='target root')
    parser.add_argument('--data_loc', type=str, default='', required=False, help='data location')
    parser.add_argument('--data_loc2', type=str, default='./modelnet40_processed/test_preprocess/xyz_chg', required=False, help='data location')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for preprocess')
    return parser.parse_args()


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def preprocess(save_path,save_path2,data_location):
    data_path = data_location
    save_path = save_path
    save_path2 = save_path2
    catfile = os.path.join(data_path, 'modelnet40_shape_names.txt')
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))
    shape_ids = {}
    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(data_path, 'modelnet40_test.txt'))]
    split = 'test'
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
    datapath = [(shape_names[i], os.path.join(data_path, shape_names[i], shape_ids[split][i]) + '.txt') for i
                        in range(len(shape_ids[split]))]
    print('The size of %s data is %d' % (split, len(datapath)))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)

    point_set_list = []
    new_xyz_list = []
    for index in tqdm(range(len(datapath))):
        fn = datapath[index]
        cls = classes[datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        point_set = point_set[0:1024, :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = point_set[:, 0:3]
        point_set = point_set[None,]
        # print(point_set.shape)
        new_point_set = torch.from_numpy(point_set)
        new_point_set = new_point_set.transpose(2, 1)
        npoint = 512
        radius = 0.2
        nsample = 32
        points = None
        new_point_set = new_point_set.permute(0, 2, 1)
        centroid = farthest_point_sample(new_point_set, npoint)
        new_xyz, new_points = sample_and_group(npoint, radius, nsample, new_point_set, points, centroid)
        
        new_xyz = new_xyz.permute(0,2,1)
        new_points = new_points.permute(0,3,2,1)
        point_set, new_xyz = new_points.numpy(),new_xyz.numpy()
        
        point_name = 'point_set'+str(index)
        if args.batch_size == 1:
            point_set.tofile(os.path.join(save_path, point_name.split('.')[0] + ".bin"))
            new_xyz.tofile(os.path.join(save_path2, point_name.split('.')[0] + ".bin"))
        else:
            point_set_list.append(point_set)
            new_xyz_list.append(new_xyz)
            if len(point_set_list) == args.batch_size:
                point_sets = np.array(point_set_list)
                new_xyzes = np.array(new_xyz_list)
                point_names = 'point_set{}.bin'.format(str(index // 16))
                point_sets.tofile(os.path.join(save_path, point_names))
                new_xyzes.tofile(os.path.join(save_path2, point_names))
                point_set_list.clear()
                new_xyz_list.clear()


def preprocess2(save_path,save_path2,data_location,data_location2):
    data_toal_folder = os.listdir(data_location)[-1]
    data_total_path = os.path.join(data_location, data_toal_folder)
    save_path = save_path
    save_path2 = save_path2
    file_start = 'point_set'
    #file_end_one = '_output_1.bin'
    #file_end_zero = '_output_0.bin'
    file_end_one = '_1.bin'
    file_end_zero = '_0.bin'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)
    test = os.path.join(data_total_path, file_start+str(0)+file_end_zero)
    try:
        file_end = file_end_zero
        test2 = np.fromfile(test,dtype=np.float32)
        test_set.shape = args.batch_size,128,512
    except:
        file_end = file_end_one
    print(file_end)    
    data_total_path2 = data_location2
    file_end_two = '.bin'
    path_file_number=glob.glob(data_location2+'/*.bin')

    for index in tqdm(range(len(path_file_number))):
        data_path2 = os.path.join(data_total_path2, file_start+str(index)+file_end_two)
        data_path1 = os.path.join(data_total_path, file_start+str(index)+file_end)
        point_set =  np.fromfile(data_path1,dtype=np.float32)
        point_set2 = np.fromfile(data_path2,dtype=np.float32)
        point_set.shape = args.batch_size,128,512
        point_set2.shape = args.batch_size,3,512
        new_point_set = torch.from_numpy(point_set2)
        point_set2 = torch.from_numpy(point_set)
        npoint = 128
        radius = 0.4
        nsample = 64
        new_point_set = new_point_set.permute(0, 2, 1)
        point_set2 = point_set2.permute(0,2,1)
        centroid = farthest_point_sample(new_point_set, npoint)
        new_xyz, new_points = sample_and_group(npoint, radius, nsample, new_point_set, point_set2, centroid)
        new_point_set = new_point_set.permute(0, 2, 1)
        new_points = new_points.permute(0,3,2,1)
        new_xyz = new_xyz.permute(0,2,1)
        point_set,new_xyz = new_points.numpy(),new_xyz.numpy()
        point_name = 'part2_'+str(index)
        point_set.tofile(os.path.join(save_path, point_name.split('.')[0] + ".bin"))
        new_xyz.tofile(os.path.join(save_path2, point_name.split('.')[0] + ".bin"))


if __name__ == '__main__':
    args = parse_args()
    if(1 == args.preprocess_part):
        preprocess(args.save_path,args.save_path2,args.data_loc)
    else:
        preprocess2(args.save_path,args.save_path2,args.data_loc,args.data_loc2)
