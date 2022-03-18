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


class SignalMonitor(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def get_signal(self):
        if self.file_path is None:
            return None
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                data = self.file.read()
                os.remove(f)
                return data
        else:
            return None
