#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#



# encoding utf-8
def hog(img, bins=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, feature_vector=True):
    """
    Extract hog feature from image.
    See detail at https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hog.py
    """
    from skimage.feature import hog
    return hog(img,
               orientations=bins,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               visualise=False,
               transform_sqrt=False,
               feature_vector=True)
