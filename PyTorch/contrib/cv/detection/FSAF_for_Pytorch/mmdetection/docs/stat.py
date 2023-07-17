# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

#!/usr/bin/env python
import os
import functools as func
import glob
import os.path as osp
import re

import numpy as np

current_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_path, '../../url.ini'), 'r') as _f:
    _content = _f.read()
    url_prefix = _content.split('prefix_url=')[1].split('\n')[0]

files = sorted(glob.glob('../configs/*/README.md'))

stats = []
titles = []
num_ckpts = 0

for f in files:
    url = osp.dirname(f.replace('../', url_prefix))

    with open(f, 'r') as content_file:
        content = content_file.read()

    title = content.split('\n')[0].replace('# ', '').strip()
    ckpts = set(x.lower().strip()
                for x in re.findall(r'\[model\]\((https?.*)\)', content))

    if len(ckpts) == 0:
        continue

    _papertype = [x for x in re.findall(r'\[([A-Z]+)\]', content)]
    assert len(_papertype) > 0
    papertype = _papertype[0]

    paper = set([(papertype, title)])

    titles.append(title)
    num_ckpts += len(ckpts)

    statsmsg = f"""
\t* [{papertype}] [{title}]({url}) ({len(ckpts)} ckpts)
"""
    stats.append((paper, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _ in stats])
msglist = '\n'.join(x for _, _, x in stats)

papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                    return_counts=True)
countstr = '\n'.join(
    [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo = f"""
# Model Zoo Statistics

* Number of papers: {len(set(titles))}
{countstr}

* Number of checkpoints: {num_ckpts}

{msglist}
"""

with open('modelzoo_statistics.md', 'w') as f:
    f.write(modelzoo)
