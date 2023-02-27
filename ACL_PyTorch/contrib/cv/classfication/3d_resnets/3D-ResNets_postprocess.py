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

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from collections import defaultdict


CLASS_NAMES = {0: 'brush_hair', 1: 'cartwheel', 2: 'catch', 3: 'chew', 4: 'clap', 5: 'climb', 
               6: 'climb_stairs', 7: 'dive', 8: 'draw_sword', 9: 'dribble', 10: 'drink', 
               11: 'eat', 12: 'fall_floor', 13: 'fencing', 14: 'flic_flac', 15: 'golf', 
               16: 'handstand', 17: 'hit', 18: 'hug', 19: 'jump', 20: 'kick', 
               21: 'kick_ball', 22: 'kiss', 23: 'laugh', 24: 'pick', 25: 'pour', 
               26: 'pullup', 27: 'punch', 28: 'push', 29: 'pushup', 30: 'ride_bike', 
               31: 'ride_horse', 32: 'run', 33: 'shake_hands', 34: 'shoot_ball', 35: 'shoot_bow', 
               36: 'shoot_gun', 37: 'sit', 38: 'situp', 39: 'smile', 40: 'smoke', 
               41: 'somersault', 42: 'stand', 43: 'swing_baseball', 44: 'sword', 45: 'sword_exercise', 
               46: 'talk', 47: 'throw', 48: 'turn', 49: 'walk', 50: 'wave'}


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def evaluate(result_path, class_names, output_topk, val_json_file):
    print('postprocessing ...')
    with open('hmdb51.info', 'r') as f:
        hmdb51_info = f.readlines()
    clip_length = {}
    for line in hmdb51_info:
        clip_id = line.split(' ')[0]
        length = int(line.split(' ')[1])
        # clips with length larger than 10 will be cut to fix batch shape
        clip_length[clip_id] = min(10, length)

    results = {'results': defaultdict(list)}

    # read bin results and batch info
    bin_list = os.listdir(result_path)

    for i in tqdm(range(len(bin_list))):
        if 'sumary.json' not in bin_list[i]:
            bin_path = os.path.join(result_path, bin_list[i])
            # bin output name format: {video_name}_output_0.bin
            video_id = bin_list[i][:-6]

            outputs = np.fromfile(bin_path, dtype=np.float32).reshape(10, 51)
            outputs = torch.from_numpy(outputs)
            outputs = F.softmax(outputs, dim=1).cpu()

            for j in range(clip_length[video_id]):
                results['results'][video_id].append({
                    'output': outputs[j]
                })

    inference_results = {'results': {}}

    for video_id, video_results in results['results'].items():
        video_outputs = [
            segment_result['output'] for segment_result in video_results
        ]
        video_outputs = torch.stack(video_outputs)
        average_scores = torch.mean(video_outputs, dim=0)
        inference_results['results'][video_id] = get_video_results(
            average_scores, class_names, output_topk)

    with open(val_json_file, 'w') as f:
        json.dump(inference_results, f)
    print("results val_json_file saved to: {}".format(val_json_file))

if __name__ == "__main__":
    evaluate(sys.argv[1], CLASS_NAMES, int(sys.argv[2]), 'val.json')
