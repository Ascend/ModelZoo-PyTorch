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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

def postprocess(src_path,save_path):
    in_files = os.listdir(src_path)
    for idx, file in enumerate(in_files):
      idx = idx + 1
      print(file, "===", idx)
      data = np.fromfile(src_path+'/'+file,np.float32).reshape(3,880,880)
      data=np.transpose(data, (1, 2, 0))
      fig=plt.gcf()
      plt.figure("visualization")
      plt.imshow(data)
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
      img_path=save_path+'/'+file.split('.')[0] + ".png"
      plt.savefig(img_path, bbox_inches='tight',dpi=300,pad_inches=0.0)
      plt.show()

if __name__ == '__main__':
    src_path = sys.argv[1]
    save_path = sys.argv[2]
    if not os.path.isdir(src_path):
        os.makedirs(os.path.realpath(src_path))
    if not os.path.isdir(save_path):
        os.makedirs(os.path.realpath(save_path))
    postprocess(src_path,save_path)
      

