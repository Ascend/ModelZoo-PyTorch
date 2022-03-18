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

'''
Runs a set of checks on a <system_desc_id>.json file.
'''

from __future__ import print_function

import argparse
import json
import sys


def _get_or_default(json, field, default):
    if field in json:
        return json[field]
    else:
        return default


def check_training_system_desc(json_file, ruleset):
    """Checks a training system desc json file for validity.

    Args:
        json_file: The system desc json file to check.
        ruleset: The ruleset such as 0.6.0 or 0.7.0.

    Returns:
        Boolean: if the system desc json is valid.
        String: system desc id.
        String: csv string ending in "," for all fields that precede results in
            summary table.
        String: csv string starting with "," for all fields that follow results
            in summary table.
    """
    with open(json_file, 'r') as f:
        contents = json.load(f)

    valid = True
    invalid_reasons = []

    required_fields = [
        "submitter",
        "divison",
        "status",
        "system_name",
        "number_of_nodes",
        "host_processors_per_node",
        "host_processor_model_name",
        "host_processor_core_count",
        "host_processor_vcpu_count",
        "host_processor_frequency",
        "host_processor_caches",
        "host_processor_interconnect",
        "host_memory_capacity",
        "host_storage_type",
        "host_storage_capacity",
        "host_networking",
        "host_networking_topology",
        "host_memory_configuration",
        "accelerators_per_node",
        "accelerator_model_name",
        "accelerator_host_interconnect",
        "accelerator_frequency",
        "accelerator_on-chip_memories",
        "accelerator_memory_configuration",
        "accelerator_memory_capacity",
        "accelerator_interconnect",
        "accelerator_interconnect_topology",
        "cooling",
        "hw_notes",
        "framework",
        "other_software_stack",
        "operating_system",
        "sw_notes",
    ]

    # Check if all required fields are contained in json.
    for field in required_fields:
        if field not in contents:
            valid = False
            invalid_reasons.append("Missing field {}".format(field))

    system_name = _get_or_default(contents, "system_name", "")

    table_csv_prefix = ",".join([
        _get_or_default(contents, "submitter", ""),
        _get_or_default(contents, "system_name", ""),
        _get_or_default(contents, "host_processor_model_name", ""),
        _get_or_default(contents, "host_processor_core_count", ""),
        _get_or_default(contents, "accelerator_model_name", ""),
        _get_or_default(contents, "accelerators_per_node", ""),
        _get_or_default(contents, "framework", ""),
    ]) + ","

    ruleset_prefix = "https://github.com/mlperf/training_results_v{}".format(ruleset)
    if "submitter" in contents and "system_name" in contents:
        details_link = "{ruleset_prefix}/blob/master/{submitter}/systems/{system_name}.json".format(
            ruleset_prefix=ruleset_prefix,
            submitter=contents["submitter"],
            system_name=contents["system_name"])
    else:
        details_link = ""
    if "submitter" in contents:
        code_link = "{ruleset_prefix}/blob/master/{submitter}/benchmarks".format(
            ruleset_prefix=ruleset_prefix,
            submitter=contents["submitter"])
    notes = ""
    table_csv_postfix = "," + ",".join([
        details_link,
        code_link,
        notes,
    ])

    if not valid:
        print("FAILURE: {}".format(", ".join(invalid_reasons)))
    else:
        print("SUCCESS")

    print("Table CSV prefix: {}".format(table_csv_prefix))
    print("Table CSV postfix: {}".format(table_csv_postfix))

    return valid, system_name, table_csv_prefix, table_csv_postfix


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.system_desc_checker',
        description='Lint MLPerf system description files.',
    )

    parser.add_argument('filename', type=str,
                    help='the file to check for compliance')
    parser.add_argument('usage', type=str,
                    help='the usage such as training, inference_edge, inference_server')
    parser.add_argument('ruleset', type=str,
                    help='the ruleset such as 0.6.0 or 0.7.0')
    parser.add_argument('--werror', action='store_true',
                    help='Treat warnings as errors')
    parser.add_argument('--quiet', action='store_true',
                    help='Suppress warnings. Does nothing if --werror is set')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.usage != 'training':
        print('Usage {} is not supported.'.format(args.usage))
        sys.exit(1)
    if args.ruleset not in ['0.6.0', '0.7.0']:
        print('Ruleset {} is not supported.'.format(args.ruleset))
        sys.exit(1)

    check_training_system_desc(args.filename, args.ruleset)


if __name__ == '__main__':
    main()
