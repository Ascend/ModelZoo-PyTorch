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
import io
from PIL import Image
try:
    import mc
except ImportError as E:
    pass


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)


class McLoader(object):

    def __init__(self, mclient_path):
        assert mclient_path is not None, \
            "Please specify 'data_mclient_path' in the config."
        self.mclient_path = mclient_path
        server_list_config_file = "{}/server_list.conf".format(
            self.mclient_path)
        client_config_file = "{}/client.conf".format(self.mclient_path)
        self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                      client_config_file)

    def __call__(self, fn):
        try:
            img_value = mc.pyvector()
            self.mclient.Get(fn, img_value)
            img_value_str = mc.ConvertBuffer(img_value)
            img = pil_loader(img_value_str)
        except:
            print('Read image failed ({})'.format(fn))
            return None
        else:
            return img