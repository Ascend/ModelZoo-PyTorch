# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script parses the dataset, extracts label and stores it at one place

Run this script from root as python src/extract_emotion_labels.py
"""

import re
import os
import pandas as pd


def extract_info():
    """
    returns info_dict containing important info from the IEMOCAP dataset
    such as start time, end time, emotion labels etc.

    extract_info: None -> Dict
    """
    info_dict = {'start_times': [], 'end_times': [], 'wav_file_names': [],
                 'emotions': [], 'vals': [], 'acts': [], 'doms': []}

    # regex used to identify useful info in the dataset files
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    for sess in range(1, 6):
        emo_evaluation_dir = 'data/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
        # Only include the session files
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir)
                            if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            # grab the important stuff
            info_lines = re.findall(info_lines, content)
            for line in info_line[1:]:  # skipping the first header line
                # Refer to the dataset to see how `line` looks like
                start_end_time, wav_file_name, emotion, val_act_dom = \
                    line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                info_dict['start_times'].append(start_time)
                info_dict['end_times'].append(end_time)
                info_dict['wav_file_names'].append(wav_file_name)
                info_dict['emotions'].append(emotion)
                info_dict['vals'].append(val)
                info_dict['acts'].append(act)
                info_dict['doms'].append(dom)
    return info_dict


def compile_dataset(info_dict):
    """
    creates a csv file from info_dict which will serve as the dataset

    compile_dataset: Dict -> None
    """
    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

    df_iemocap['start_time'] = info_dict['start_times']
    df_iemocap['end_time'] = info_dict['end_times']
    df_iemocap['wav_file'] = info_dict['wav_file_names']
    df_iemocap['emotion'] = info_dict['emotions']
    df_iemocap['val'] = info_dict['vals']
    df_iemocap['act'] = info_dict['acts']
    df_iemocap['dom'] = info_dict['doms']
    # Finally, save to a file
    df_iemocap.to_csv('data/pre-processed/df_iemocap.csv', index=False)


def main():
    compile_dataset(extract_info())


if __name__ == '__main__':
    main()
