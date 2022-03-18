# Copyright 2020 Huawei Technologies Co., Ltd## Licensed under the Apache License, Version 2.0 (the "License");# you may not use this file except in compliance with the License.# You may obtain a copy of the License at## http://www.apache.org/licenses/LICENSE-2.0## Unless required by applicable law or agreed to in writing, software# distributed under the License is distributed on an "AS IS" BASIS,# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.# See the License for the specific language governing permissions and# limitations under the License.# ============================================================================import ctypes
import random
import string


def get_random_string(length=15):
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Default: 15.
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id():
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir():
    """Get shm dir for temporary usage."""
    return '/dev/shm'
