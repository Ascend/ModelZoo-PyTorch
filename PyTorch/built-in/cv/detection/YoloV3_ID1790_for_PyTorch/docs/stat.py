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

#!/usr/bin/env python
import glob
import os.path as osp
import re

url_prefix = 'https://github.com/open-mmlab/mmdetection/blob/master/'

files = sorted(glob.glob('../configs/*/README.md'))

stats = []
titles = []
num_ckpts = 0

for f in files:
    url = osp.dirname(f.replace('../', url_prefix))

    with open(f, 'r') as content_file:
        content = content_file.read()

    title = content.split('\n')[0].replace('# ', '')

    titles.append(title)
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https?://download.*\.pth', content)
                if 'mmdetection' in x)
    num_ckpts += len(ckpts)
    statsmsg = f"""
\t* [{title}]({url}) ({len(ckpts)} ckpts)
"""
    stats.append((title, ckpts, statsmsg))

msglist = '\n'.join(x for _, _, x in stats)

modelzoo = f"""
# Model Zoo Statistics

* Number of papers: {len(titles)}
* Number of checkpoints: {num_ckpts}
{msglist}
"""

with open('modelzoo_statistics.md', 'w') as f:
    f.write(modelzoo)
