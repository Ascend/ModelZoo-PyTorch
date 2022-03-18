# Copyright 2019 MLBenchmark Group. All Rights Reserved.
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
import sys

from . import mlp_compliance


parser = mlp_compliance.get_parser()
args = parser.parse_args()

config_file = args.config or f'{args.ruleset}/common.yaml'

checker = mlp_compliance.make_checker(
    args.ruleset,
    args.quiet,
    args.werror,
)

valid, system_id, benchmark, result = mlp_compliance.main(args.filename, config_file, checker)

print(valid)
print(system_id)
print(benchmark)
print(result)

if not valid:
    sys.exit(1)
