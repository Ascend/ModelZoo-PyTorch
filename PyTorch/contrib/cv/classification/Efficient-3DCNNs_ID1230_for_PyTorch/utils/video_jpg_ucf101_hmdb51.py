# Copyright 2021 Huawei Technologies Co., Ltd
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
#coding=utf-8
from __future__ import print_function, division
import os
import sys
import subprocess

def class_process(dir_path, dst_dir_path, class_name):
  class_path = os.path.join(dir_path, class_name)
  if not os.path.isdir(class_path):
    return

  dst_class_path = os.path.join(dst_dir_path, class_name)
  if not os.path.exists(dst_class_path):
    os.mkdir(dst_class_path)
    # os.makedirs(dst_class_path)

  for file_name in os.listdir(class_path):
    if '.avi' not in file_name:
      continue
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)
    try:
      if os.path.exists(dst_directory_path):
        if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
          subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
          print('remove {}'.format(dst_directory_path))
          os.mkdir(dst_directory_path)
        else:
          continue
      else:
        os.mkdir(dst_directory_path)
    except:
      print(dst_directory_path)
      continue

    cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')

if __name__=="__main__":
  # dir_path = sys.argv[1]
  # dst_dir_path = sys.argv[2]
  dir_path = '../annotation_UCF101/UCF-101/'
  dst_dir_path = '../annotation_UCF101/UCF-101-image/'

  for class_name in os.listdir(dir_path):
    class_process(dir_path, dst_dir_path, class_name)
