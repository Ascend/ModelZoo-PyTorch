# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
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
#
#
# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from detectron2.utils.collect_env import collect_env_info


class TestProjects(unittest.TestCase):
    def test_import(self):
        from detectron2.projects import point_rend

        _ = point_rend.add_pointrend_config

        import detectron2.projects.deeplab as deeplab

        _ = deeplab.add_deeplab_config

        # import detectron2.projects.panoptic_deeplab as panoptic_deeplab

        # _ = panoptic_deeplab.add_panoptic_deeplab_config


class TestCollectEnv(unittest.TestCase):
    def test(self):
        _ = collect_env_info()
