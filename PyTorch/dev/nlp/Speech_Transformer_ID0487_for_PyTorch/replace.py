# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
# -*- coding: utf-8 -*-
import json

if __name__ == '__main__':
    with open('README.t', 'r', encoding="utf-8") as file:
        text = file.readlines()
    text = ''.join(text)

    with open('results.json', 'r', encoding="utf-8") as file:
        results = json.load(file)

    print(results[0])

    for i, result in enumerate(results):
        out_key = 'out_list_{}'.format(i)
        text = text.replace('$({})'.format(out_key), '<br>'.join(result[out_key]))
        gt_key = 'gt_{}'.format(i)
        text = text.replace('$({})'.format(gt_key), result[gt_key])

    text = text.replace('<sos>', '')
    text = text.replace('<eos>', '')

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(text)
