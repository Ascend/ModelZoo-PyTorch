# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#--------------------------------------------------------------------
# modified from "ADVENT/advent/dataset/cityscapes.py" by Tuan-Hung Vu
#--------------------------------------------------------------------
import numpy as np

from advent.utils import project_root
from advent.utils.serialization import json_load
from advent.dataset.base_dataset import BaseDataset

DEFAULT_INFO_PATH = project_root / 'advent/dataset/cityscapes_list/info.json'


class CityscapesDataSet(BaseDataset):
    def __init__(self, root, list_path, set='train',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True, labels_size=None):
        # pdb.set_trace()
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels
        # self.info = json_load(info_path)
        # self.mapping = np.array(self.info['label2train'], dtype=np.int)
        # self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        # for source_label, target_label in self.mapping:
            # self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        # label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        # pdb.set_trace()
        label_name = name.split('/')[1]
        label_file = '../entropy_rank/color_masks/' + label_name 
        # label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    # def map_labels(self, input_):
    #     return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        # label = self.map_labels(label).copy()
        image = self.get_image(img_file)
        image = self.preprocess(image)
        return image.copy(), label, np.array(image.shape), name
