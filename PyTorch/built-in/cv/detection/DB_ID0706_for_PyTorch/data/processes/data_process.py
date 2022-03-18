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

from concern.config import Configurable


class DataProcess(Configurable):
    r'''Processes of data dict.
    '''

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        raise NotImplementedError

    def render_constant(self, canvas, xmin, xmax, ymin, ymax, value=1, shrink=0):
        def shrink_rect(xmin, xmax, ratio):
            center = (xmin + xmax) / 2
            width = center - xmin
            return int(center - width * ratio + 0.5), int(center + width * ratio + 0.5)

        if shrink > 0:
            xmin, xmax = shrink_rect(xmin, xmax, shrink)
            ymin, ymax = shrink_rect(ymin, ymax, shrink)

        canvas[int(ymin+0.5):int(ymax+0.5)+1, int(xmin+0.5):int(xmax+0.5)+1] = value
        return canvas
