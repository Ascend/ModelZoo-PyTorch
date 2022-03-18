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

import re
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-step_start", default=1300, type=int)
    parser.add_argument("-step_end", default=1400, type=int)
    parser.add_argument("-log_path", default='.')
    parser.add_argument("-log_file", default="log.train_performance_1p.txt")
    

    args = parser.parse_args()
    
    filename = os.path.join(args.log_path, args.log_file)
    pattern_start = "Step " + str(args.step_start) + "/"
    pattern_end = "Step " + str(args.step_end)
    
    with open(file=filename, mode='r') as f:
        for text in f:
            if re.search(pattern_start, text):
                print(text)
                start_time = text.split(" ")[-2]
            if re.search(pattern_end, text):
                print(text)
                end_time = text.split(" ")[-2]
                break
        
    iter_time = (float(end_time) - float(start_time)) / (args.step_end - args.step_start)
    print("average iter_time from Step %d to Step %d: %5.0fms" % (args.step_start, args.step_end, iter_time*1000))
