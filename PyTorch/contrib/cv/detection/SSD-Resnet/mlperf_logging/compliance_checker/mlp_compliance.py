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
Runs the set of rules defined in provided config YAML file.
'''

from __future__ import print_function

import argparse
import os
import yaml
import json
import re
import math

from . import mlp_parser


def is_integer(value):
    return abs(round(value) - value) < 0.00001


class CCError(Exception): 
    pass


def preety_dict(d):
    return json.dumps(d, sort_keys=True, indent=2)


enqueued_configs = []

# this function can be called from yaml
def enqueue_config(config):
    enqueued_configs.append(config)


def all_same(l):
    return not l or len(l) == l.count(l[0])

def merge(*dicts,):
    return { k:d[k] for d in dicts for k in d }

class ComplianceChecker:

    def __init__(self, ruleset, quiet, werror):
        self.ruleset = ruleset

        self.warnings = {}
        self.overwritable = {}
        self.not_overwritable = []

        self.quiet = quiet
        self.werror = werror

    def raise_exception(self, msg):
        raise CCError(msg)


    def put_warning(self, msg, key):
        if self.werror:
            self.put_message(msg, key)
        elif not self.quiet:
            print(key, msg)
            self.warnings[key] = msg

    def put_message(self, msg, key=None):
        if key:
            self.overwritable[key] = msg
        else:
            self.not_overwritable.append(msg)


    def overwrite_messages(self, keys):
        for key in keys:
            self.overwritable.pop(key, None)


    def log_messages(self):
        message_separator = '\n' + '-' * 30 + '\n'
        message = message_separator.join([
                    *self.warnings.values(),
                    *self.overwritable.values(),
                    *self.not_overwritable
        ])
        if message:
          print(message)

    def has_messages(self):
        return self.not_overwritable or self.overwritable


    def run_check_eval(self, ll, tests, state):
        if type(tests) is not list:
            tests = [tests]

        for test in tests:
            try:
                if not eval(test.strip(), state, {'ll': ll, 'v': ll.value }):
                    self.put_message(
                        f"CHECK for '{ll.key}' failed in line {ll.lineno}:"
                        f"\n{ll.full_string}"
                        f"\nfailed test: {test}"
                        f"\ncurrent context[s]={preety_dict(state['s'])}"
                        f"\ncurrent line[v]={preety_dict(ll.value)}",
                        key=ll.key
                    )
            except:
                self.put_message(
                    f'Failed executing CHECK code:'
                    f'\n{test}'
                    f'\ntriggered by line:'
                    f'\n{ll.full_string}'
                    f"\ncurrent context[s]={preety_dict(state['s'])}",
                    key=ll.key
                )

    def run_check_end(self, tests, state):
        if type(tests) is not list:
            tests = [tests]

        for test in tests:
            try:
                if not eval(test.strip(), state):
                    self.put_message(
                        f"failed test: {test}"
                        f"\ncurrent context[s]={preety_dict(state['s'])}",
                    )
            except:
                self.put_message(
                    f'Failed executing CHECK code:'
                    f'\n{test}'
                    f'\ncurrent context[s]={preety_dict(state["s"])}'
                )

    def run_check_exec(self, ll, code, state, action):
        if code is None: return

        try:
            exec(code.strip(), state, {'ll': ll, 'v': ll.value})
        except:
            self.put_message(f'Failed executing code {action} code triggered by line :\n{ll.full_string}',
                             key=ll.key)


    def parse_alternatives(self, string):
        in_pharentises = string[len('AT_LEAST_ONE_OR(') : -1]
        alternatives = in_pharentises.split(',')
        return [s.strip() for s in alternatives]

    def configured_checks(self, loglines, config_file):
        with open(config_file) as f:
            checks = yaml.load(f, Loader=yaml.BaseLoader)

        if checks is None:
            return

        s = {}  # this would be visible from inside configs
        state = {'enqueue_config':enqueue_config , 's':s,
         'is_integer': is_integer,
         'math': math}


        #execute begin block
        begin_blocks = [x for x in checks if list(x)[0]=='BEGIN']
        assert(len(begin_blocks)<=1) # up to one begin block
        if len(begin_blocks)==1:
            exec(begin_blocks[0]['BEGIN']['CODE'].strip(), state)

        key_records = {}
        for k in checks:
            if list(k)[0]=='KEY':
                key_records.update({k['KEY']['NAME']:k['KEY']}) 

        reported_values = {k:[] for k in key_records.keys()}

        # if config overrides some rules from previous config, corresponding messages are not needed
        self.overwrite_messages(key_records)

        at_least_one_checks = {}
        # executing the rules through log records
        for line in loglines:
            key_record = None
            try:
                reported_values[line.key].append(line.value['value'])
                key_record = key_records[line.key]
            except:
                # unknown key - it's allowed, skip to next record
                continue

            if 'PRE' in key_record: self.run_check_exec(line, key_record['PRE'], state, 'PRE')
            if 'CHECK' in key_record: self.run_check_eval(line, key_record['CHECK'], state)
            if 'POST' in key_record: self.run_check_exec(line, key_record['POST'], state, 'POST')
            if 'ATLEAST_ONE_CHECK' in key_record:
              if line.key not in at_least_one_checks:
                at_least_one_checks[line.key] = [0, key_record['ATLEAST_ONE_CHECK']]
              check = eval(key_record['ATLEAST_ONE_CHECK'].strip(),
                           state, {'ll': line, 'v': line.value})
              if check:
                at_least_one_checks[line.key][0] += 1
        for name in at_least_one_checks:
          if at_least_one_checks[name][0] == 0:
            self.put_message('Failed checks for {} : {}'
                             .format(name, at_least_one_checks[name][1]))


        alternatives = set()
        # verify occurrences requirements
        for k,v in key_records.items():
            if 'REQ' not in v:
                continue

            if v['REQ']=='EXACTLY_ONE':
                if len(reported_values[k]) !=1:
                    if reported_values[k] and all_same(reported_values[k]):
                        self.put_warning(f"Required EXACTLY_ONE occurrence of '{k}'"
                                         f" but found {len(reported_values[k])}",
                                         key=k)
                    else:
                        self.put_message(f"Required EXACTLY_ONE occurrence of '{k}'"
                                         f" but found {len(reported_values[k])}" +
                                         (f" with different values:"
                                         f" [{', '.join(x for x in set(str(v) for v in reported_values[k]))}]"
                                         if reported_values[k] else ''),
                                         key=k)

            if v['REQ']=='AT_LEAST_ONE':
                if len(reported_values[k])<1:
                     self.put_message(f"Required AT_LEAST_ONE occurrence of '{k}' but found {len(reported_values[k])}",
                                      key=k)

            if v['REQ'].startswith('AT_LEAST_ONE_OR'):
                alternatives.add(tuple({k, *self.parse_alternatives(v['REQ'])}))

        for alts in alternatives:
            if not any(reported_values[k] for k in alts):
                self.put_message("Required AT_LEAST_ONE occurrence of {}".format(' or '.join(f"'{s}'" for s in alts)))

        # execute end block
        end_blocks = [x for x in checks if list(x)[0]=='END']
        assert(len(end_blocks)<=1) # up to one end block
        if len(end_blocks)==1:
            end_record = end_blocks[0]['END']
            if 'PRE' in end_record: exec(end_record['PRE'].strip(), state)
            if 'CHECK' in end_record:
                self.run_check_end(end_record['CHECK'], state)

    def check_loglines(self, loglines, config):
        if not loglines:
          self.put_message('No log lines detected')

        enqueue_config(config)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        while len(enqueued_configs)>0:
            current_config = enqueued_configs.pop(0)
            config_file = general_file = os.path.join(current_dir, current_config)

            if not os.path.exists(config_file):
                self.put_message('Could not find config file: {}'.format(config_file))

            # processing a config may have a side affect of pushing another config(s) to be checked
            self.configured_checks(loglines,  config_file)


    def check_file(self, filename, config_file):

        loglines, errors = mlp_parser.parse_file(filename, ruleset=self.ruleset)

        if len(errors) > 0:
            print('Found parsing errors:')
            for line, error in errors:
                print(line)
                print('  ^^ ', error)
            print()
            self.put_message('Log lines had parsing errors.')

        self.check_loglines(loglines, config_file)

        self.log_messages()

        return not self.has_messages()


def rule_choices():
    return [ x for x in os.listdir(os.path.dirname(__file__))
            if re.match('\d+\.\d+\.\d+', x) ]


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.compliance_checker',
        description='Lint MLPerf Compliance Logs.',
    )

    parser.add_argument('filename', type=str,
                    help='the file to check for compliance')
    parser.add_argument('--ruleset', type=str, default='0.7.0',
                    choices=rule_choices(),
                    help='what version of rules to check the log against')
    parser.add_argument('--config',  type=str,
                    help='mlperf logging config, by default it loads {ruleset}/common.yaml', default=None)
    parser.add_argument('--werror', action='store_true',
                    help='Treas warnings as errors')
    parser.add_argument('--quiet', action='store_true',
                    help='Suppress warnings. Does nothing if --werror is set')

    return parser


def make_checker(ruleset, quiet, werror):
    return ComplianceChecker(
        ruleset,
        quiet,
        werror,
    )


def main(filename, config_file, checker):
    valid = checker.check_file(filename, config_file)

    return valid, None, None, None

