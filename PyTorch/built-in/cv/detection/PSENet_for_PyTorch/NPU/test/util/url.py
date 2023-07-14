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



import os
import sys

from six.moves import urllib

import util


def download(url, path):
    filename = path.split('/')[-1]
    if not util.io.exists(path):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r-----Downloading %s %.1f%%' % (filename,
                                                               float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        path, _ = urllib.request.urlretrieve(url, path, _progress)
        print()
        statinfo = os.stat(path)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def get_url(name):
    if not name:
        return ""
    with open('./url.ini', 'r') as f:
        content = f.read()
        if name not in content:
            return ""
        img_url = content.split(name+'=')[1].split('\n')[0]
    return img_url

