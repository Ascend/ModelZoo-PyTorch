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

from PIL import Image
import cv2
import base64
import io
import numpy as np


def convert(data):
    if isinstance(data, dict):
        ndata = {}
        for key, value in data.items():
            nkey = key.decode()
            if nkey == 'img':
                img = Image.open(io.BytesIO(value))
                img = img.convert('RGB')
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                nvalue = img
            else:
                nvalue = convert(value)
            ndata[nkey] = nvalue
        return ndata
    elif isinstance(data, list):
        return [convert(item) for item in data]
    elif isinstance(data, bytes):
        return data.decode()
    else:
        return data


def to_np(x):
    return x.cpu().data.numpy()
