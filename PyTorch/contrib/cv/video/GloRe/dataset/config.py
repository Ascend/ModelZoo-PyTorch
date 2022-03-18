# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

import logging

def get_config(name):

    config = {}

    if name.upper() == 'UCF101':
        config['num_classes'] = 101
    elif name.upper() == 'HMDB51':
        config['num_classes'] = 51
    elif name.upper() == 'KINETICS':
        config['num_classes'] = 400
    else:
        logging.error("Configs for dataset '{}'' not found".format(name))
        raise NotImplemented

    logging.debug("Target dataset: '{}', configs: {}".format(name.upper(), config))

    return config


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info(get_config("ucf101"))
    logging.info(get_config("HMDB51"))
    logging.info(get_config("Kinetics"))
