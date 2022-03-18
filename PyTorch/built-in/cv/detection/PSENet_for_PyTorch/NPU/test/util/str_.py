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



# encoding = utf-8
def int_array_to_str(arr):
    """turn an int array to a str"""
    return "".join(map(chr, arr))


def join(arr, splitter=','):
    temp = []
    for e in arr:
        temp.append(e)
        temp.append(splitter)
    temp.pop()
    return "".join(temp)


def is_str(s):
    return type(s) == str


def to_lowercase(s):
    return str.lower(s)


def to_uppercase(s):
    return str.upper(s)


def ends_with(s, suffix, ignore_case=False):
    """
    suffix: str, list, or tuple
    """
    if is_str(suffix):
        suffix = [suffix]
    suffix = list(suffix)
    if ignore_case:
        for idx, suf in enumerate(suffix):
            suffix[idx] = to_lowercase(suf)
        s = to_lowercase(s)
    suffix = tuple(suffix)
    return s.endswith(suffix)


def starts_with(s, prefix, ignore_case=False):
    """
    prefix: str, list, or tuple
    """
    if is_str(prefix):
        prefix = [prefix]
    prefix = list(prefix)
    if ignore_case:
        for idx, pre in enumerate(prefix):
            prefix[idx] = to_lowercase(pre)
        s = to_lowercase(s)
    prefix = tuple(prefix)
    return s.startswith(prefix)


def contains(s, target, ignore_case=False):
    if ignore_case:
        s = to_lowercase(s)
        target = to_lowercase(target)
    return s.find(target) >= 0


def index_of(s, target):
    return s.find(target)


def replace_all(s, old, new, reg=False):
    if reg:
        import re
        targets = re.findall(old, s)
        for t in targets:
            s = s.replace(t, new)
    else:
        s = s.replace(old, new)
    return s


def remove_all(s, sub):
    return replace_all(s, sub, '')


def split(s, splitter, reg=False):
    if not reg:
        return s.split(splitter)
    import re
    return re.split(splitter, s)


def remove_invisible(s):
    s = replace_all(s, ' ', '')
    s = replace_all(s, '\n', '')
    s = replace_all(s, '\t', '')
    s = replace_all(s, '\r', '')
    return s


def find_all(s, pattern):
    import re
    return re.findall(pattern, s)


def is_none_or_empty(s):
    if s is None:
        return True
    return len(s) == 0;
