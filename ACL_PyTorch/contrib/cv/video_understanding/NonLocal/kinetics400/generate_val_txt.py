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
import os

import cv2
import csv


def check_mp4_file(path):
    cap = cv2.VideoCapture(path)
    res, frame = cap.read()

    return res


def generate_txt(dataset_root, val_csv, video_file, val_txt):
    label2number = {}
    val_dict = {}
    num = -1
    with open(os.path.join(dataset_root, val_csv), 'r') as csv_file:
        for index_row, row in enumerate(csv.reader(csv_file)):
            if index_row > 0:
                if len(row[0].split(' ')) > 1:
                    label = ''
                    for index_row0, item in enumerate(row[0].split(' ')):
                        if index_row0 > 0:
                            label += '_'
                        label += item
                else:
                    label = row[0]
                
                if label not in label2number.keys():
                    num += 1
                    label2number[label] = num
                val_dict[row[1]] = label2number[label]
    
    txt_file = open(os.path.join(dataset_root, val_txt), 'w+')
    video_root = os.path.join(dataset_root, video_file)
    for video in os.listdir(video_root):
        if check_mp4_file(os.path.join(video_root, video)):
            try:
                txt_file.writelines([video, ' ', str(val_dict[video.split('_00')[0]])])
            except KeyError:
                txt_file.writelines([video, ' ', str(val_dict[video.split('_000')[0]])])
            txt_file.writelines('\n')
    txt_file.close()


if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='Generate txt file from kinetics validation dataset')
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--val_csv', type=str, default=None)
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--val_txt', type=str, default=None)
    args = parser.parse_args()
    generate_txt(args.dataset_root, args.val_csv, args.video_file, args.val_txt)
    