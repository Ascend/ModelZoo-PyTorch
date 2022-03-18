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

from concern.config import State

from .data_process import DataProcess


class FilterKeys(DataProcess):
    required = State(default=[])
    superfluous = State(default=[])

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        self.required_keys = set(self.required)
        self.superfluous_keys = set(self.superfluous)
        if len(self.required_keys) > 0 and len(self.superfluous_keys) > 0:
            raise ValueError(
                'required_keys and superfluous_keys can not be specified at the same time.')

    def process(self, data):
        for key in self.required:
            assert key in data, '%s is required in data' % key

        superfluous = self.superfluous_keys
        if len(superfluous) == 0:
            for key in data.keys():
                if key not in self.required_keys:
                    superfluous.add(key)

        for key in superfluous:
            del data[key]
        return data
