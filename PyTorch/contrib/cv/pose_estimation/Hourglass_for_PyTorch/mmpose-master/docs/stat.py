# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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

#!/usr/bin/env python
import functools as func
import glob
import re

files = sorted(glob.glob('*_models.md'))

stats = []

for f in files:
    with open(f, 'r') as content_file:
        content = content_file.read()

    # title
    title = content.split('\n')[0].replace('#', '')

    # count papers
    papers = set(x.lower().strip()
                 for x in re.findall(r'\btitle={(.*)}', content))
    paperlist = '\n'.join(sorted('    - ' + x for x in papers))
    # count configs
    configs = set(x.lower().strip()
                  for x in re.findall(r'https.*configs/.*\.py', content))

    # count ckpts
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https://download.*\.pth', content)
                if 'mmpose' in x)

    statsmsg = f"""
## [{title}]({f})

* Number of checkpoints: {len(ckpts)}
* Number of configs: {len(configs)}
* Number of papers: {len(papers)}
{paperlist}

    """

    stats.append((papers, configs, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _, _ in stats])
allconfigs = func.reduce(lambda a, b: a.union(b), [c for _, c, _, _ in stats])
allckpts = func.reduce(lambda a, b: a.union(b), [c for _, _, c, _ in stats])
msglist = '\n'.join(x for _, _, _, x in stats)

modelzoo = f"""
# Model Zoo Statistics

* Number of checkpoints: {len(allckpts)}
* Number of configs: {len(allconfigs)}
* Number of papers: {len(allpapers)}

{msglist}

"""

with open('modelzoo.md', 'w') as f:
    f.write(modelzoo)
