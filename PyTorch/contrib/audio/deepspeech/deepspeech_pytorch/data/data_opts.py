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

def add_data_opts(parser):
    data_opts = parser.add_argument_group("General Data Options")
    data_opts.add_argument('--manifest-dir', default='./', type=str,
                           help='Output directory for manifests')
    data_opts.add_argument('--min-duration', default=1, type=int,
                           help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
    data_opts.add_argument('--max-duration', default=15, type=int,
                           help='Prunes training samples longer than the max duration (given in seconds, default 15)')
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
    return parser
