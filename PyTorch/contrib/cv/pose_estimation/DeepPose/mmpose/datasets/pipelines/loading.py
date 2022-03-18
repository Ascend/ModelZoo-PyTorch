# Copyright 2020 Huawei Technologies Co., Ltd
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
import mmcv

from ..registry import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        image_file = results['image_file']
        img = mmcv.imread(image_file, self.color_type, self.channel_order)

        if img is None:
            raise ValueError('Fail to read {}'.format(image_file))

        results['img'] = img
        return results
