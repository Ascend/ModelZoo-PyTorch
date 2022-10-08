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
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append(r"./Efficient-3DCNNs/utils")
from eval_ucf101 import UCFclassification

CLASS_NAMES = {0: 'ApplyEyeMakeup', 1: 'ApplyLipstick', 2: 'Archery', 3: 'BabyCrawling', 4: 'BalanceBeam', 5: 'BandMarching',
               6: 'BaseballPitch', 7: 'Basketball', 8: 'BasketballDunk', 9: 'BenchPress', 10: 'Biking',
               11: 'Billiards', 12: 'BlowDryHair', 13: 'BlowingCandles', 14: 'BodyWeightSquats', 15: 'Bowling',
               16: 'BoxingPunchingBag', 17: 'BoxingSpeedBag', 18: 'BreastStroke', 19: 'BrushingTeeth', 20: 'CleanAndJerk',
               21: 'CliffDiving', 22: 'CricketBowling', 23: 'CricketShot', 24: 'CuttingInKitchen', 25: 'Diving',
               26: 'Drumming', 27: 'Fencing', 28: 'FieldHockeyPenalty', 29: 'FloorGymnastics', 30: 'FrisbeeCatch',
               31: 'FrontCrawl', 32: 'GolfSwing', 33: 'Haircut', 34: 'Hammering', 35: 'HammerThrow',
               36: 'HandstandPushups', 37: 'HandstandWalking', 38: 'HeadMassage', 39: 'HighJump', 40: 'HorseRace',
               41: 'HorseRiding', 42: 'HulaHoop', 43: 'IceDancing', 44: 'JavelinThrow', 45: 'JugglingBalls',
               46: 'JumpingJack', 47: 'JumpRope', 48: 'Kayaking', 49: 'Knitting', 50: 'LongJump',
               51: 'Lunges', 52: 'MilitaryParade', 53: 'Mixing', 54: 'MoppingFloor', 55: 'Nunchucks',
               56: 'ParallelBars', 57: 'PizzaTossing', 58: 'PlayingCello', 59: 'PlayingDaf', 60: 'PlayingDhol',
               61: 'PlayingFlute', 62: 'PlayingGuitar', 63: 'PlayingPiano', 64: 'PlayingSitar', 65: 'PlayingTabla',
               66: 'PlayingViolin', 67: 'PoleVault', 68: 'PommelHorse', 69: 'PullUps', 70: 'Punch',
               71: 'PushUps', 72: 'Rafting', 73: 'RockClimbingIndoor', 74: 'RopeClimbing', 75: 'Rowing',
               76: 'SalsaSpin', 77: 'ShavingBeard', 78: 'Shotput', 79: 'SkateBoarding', 80: 'Skiing',
               81: 'Skijet', 82: 'SkyDiving', 83: 'SoccerJuggling', 84: 'SoccerPenalty', 85: 'StillRings',
               86: 'SumoWrestling', 87: 'Surfing', 88: 'Swing', 89: 'TableTennisShot', 90: 'TaiChi',
               91: 'TennisSwing', 92: 'ThrowDiscus', 93: 'TrampolineJumping', 94: 'Typing', 95: 'UnevenBars',
               96: 'VolleyballSpiking', 97: 'WalkingWithDog', 98: 'WallPushups', 99: 'WritingOnBoard', 100: 'YoYo'}

def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[int(locs[i])],
            'score': float(sorted_scores[i])
        })

    test_results['results'][video_id] = video_results

def evaluate(result_path, class_names, info_path, annotation_path, acc_file):
    print('postprocessing')
    f = open(info_path, 'r')
    ucf101_info = f.readlines()
    bin_list = os.listdir(result_path)
    bin_list.remove('sumary.json')
    bin_list.sort(key= lambda x:int(x[:-6]))
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, line in enumerate(ucf101_info):
        targets = line.split(' ')
        targets = targets[0:len(targets)-1]
        bin_path = os.path.join(result_path, bin_list[i])
        outputs = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 101)
        outputs = torch.from_numpy(outputs)
        outputs = F.softmax(outputs, dim=1).cpu()
        
        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]



        if (i % 100) == 0:
            with open('val.json', 'w') as f:
                json.dump(test_results, f)
        if (i % 1000) == 0:
             print('[{}/{}]'.format(i+1, len(bin_list)))
    with open('val.json', 'w') as f:
        json.dump(test_results, f)

    ucf_acc_t1 = UCFclassification(annotation_path, 'val.json', subset='validation', top_k=1)
    ucf_acc_t1.evaluate()

    ucf_acc_t5 = UCFclassification(annotation_path, 'val.json', subset='validation', top_k=5)
    ucf_acc_t5.evaluate()

    with open(acc_file, 'w') as f:
        json.dump('top1 acc:'+str(ucf_acc_t1.hit_at_k)+'; top5 acc:'+str(ucf_acc_t5.hit_at_k), f)
    print('postprocess finised')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of 3D-ResNets')
    parser.add_argument('--result_path', default='', type=Path, help='Directory path of videos')
    parser.add_argument('--info_path', default='', type=Path, help='Directory path of binary output data')
    parser.add_argument('--annotation_path', default='', type=Path, help='Annotation file path')
    parser.add_argument('--acc_file', default='', type=Path, help='Directory path of binary output data')
    opt = parser.parse_args()
    evaluate(opt.result_path, CLASS_NAMES, opt.info_path, opt.annotation_path, opt.acc_file)
