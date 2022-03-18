#     Copyright 2021 Huawei Technologies Co., Ltd
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

import platform


def get_torchaudio_backend():
    """Get the backend for torchaudio between soundfile and sox_io according to the os.

    Allow users to use soundfile or sox_io according to their os.

    Returns
    -------
    str
        The torchaudio backend to use.
    """
    current_system = platform.system()
    if current_system == "Windows":
        return "soundfile"
    else:
        return "sox"
