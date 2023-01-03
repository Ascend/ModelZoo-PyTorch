# Copyright 2022 Huawei Technologies Co., Ltd
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
from codes.utils import scandir
from codes.utils.lmdb_util import make_lmdb_from_imgs
import ascend_function


def create_lmdb():

    folder_path = 'datasets/FiveK/FiveK_480p/train/A'
    lmdb_path = 'datasets/FiveK/FiveK_train_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/FiveK/FiveK_480p/train/B'
    lmdb_path = 'datasets/FiveK/FiveK_train_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/FiveK/FiveK_480p/test/A'
    lmdb_path = 'datasets/FiveK/FiveK_test_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/FiveK/FiveK_480p/test/B'
    lmdb_path = 'datasets/FiveK/FiveK_test_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def prepare_keys(folder_path):

    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='jpg', recursive=False)))
    keys = [img_path.split('.jpg')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == '__main__':

    create_lmdb()

