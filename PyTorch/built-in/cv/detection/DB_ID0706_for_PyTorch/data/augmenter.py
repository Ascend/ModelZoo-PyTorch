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

import imgaug
import imgaug.augmenters as iaa

from concern.config import Configurable, State


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None:
            return None
        elif isinstance(args, (int, float, str)):
            return args
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(
                    *[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            if 'cls' in args:
                cls = getattr(iaa, args['cls'])
                return cls(**{k: self.to_tuple_if_list(v) for k, v in args.items() if not k == 'cls'})
            else:
                return {key: self.build(value, root=False) for key, value in args.items()}
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj
