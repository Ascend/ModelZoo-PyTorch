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
import sys
import argparse
import json
import os
import torch
import torch.utils.data as data
import json
from GloRe.data import video_sampler as sampler
from GloRe.data import video_transforms as transforms
from GloRe.data.video_iterator import VideoIter

def get_info(input_path,output_path):
    if not os.path.exists('class_info.json'):
        class_dict = {}
        data_dir = input_path
        class_list = os.listdir(data_dir)
        class_list.sort()
        count = 0
        label = 0
        f = open(output_path, 'w')
        for class_label in class_list:
            class_path = os.path.join(data_dir, class_label)
            if os.path.exists(class_path):
                av_list = os.listdir(class_path)
                for av in av_list:
                    f.write(str(count) + ' ' + str(label) + ' ' + class_label + '/' + av + '\n')
                    count += 1
                class_dict[str(label)] = class_label
                label += 1
        print(count)
        f.close()
        with open('class_info.json','w') as f:
            json.dump(class_dict,f)

    else:
        with open('class_info.json', 'r') as f:
            class_dict = json.load(f)
        data_dir = input_path
        count = 0
        label = 0
        f = open(output_path,'w')
        for key,class_label in class_dict.items():
            class_path = os.path.join(data_dir, class_label)
            if os.path.exists(class_path):
                av_list = os.listdir(class_path)
                for av in av_list:
                    f.write(str(count)+' '+str(key)+' '+class_label+'/'+str(av)+'\n')
                    count += 1
                label += 1
        print(count)
        f.close()

parser = argparse.ArgumentParser(description="GloRe Preprocess")
# io
parser.add_argument('--clip-length', default=8,
                    help="define the length of each input sample.")
parser.add_argument('--frame-interval', type=int, default=8,
                    help="define the sampling interval between frames.")
parser.add_argument('--data-root', type=str, default='dataset/UCF101',
                    help="data root path")

parser.add_argument('--save-path', type=str, default='bin/UCF101',
                    help="save bin path")
parser.add_argument('--json-path', type=str, default='target',
                    help="save res json path")
# evaluation
parser.add_argument('--batch-size', type=int, default=16,
                    help="batch size")
parser.add_argument('--txt-list', type=str, default='datalist.txt', help='video txt list in data root path')

if __name__ == '__main__':
    args = parser.parse_args()
    data_root = args.data_root
    save_path = args.save_path
    get_info(data_root,args.txt_list)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    input_config = {}
    input_config['mean'] = [124 / 255, 117 / 255, 104 / 255]
    input_config['std'] = [1 / (.0167 * 255)] * 3
    num_clips = 1
    num_crops = 1
    normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
    val_sampler = sampler.EvenlySampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         num_times=num_clips)
    val_loader = VideoIter(video_prefix=os.path.join(data_root),
                           txt_list=os.path.join(args.txt_list),
                           sampler=val_sampler,
                           force_color=True,
                           video_transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop((224, 224)),
                               transforms.ToTensor(),
                               normalize,
                           ]),
                           name='test',
                           return_item_subpath=True,
                           list_repeat_times=(num_clips * num_crops),
                           )

    inference_loader = torch.utils.data.DataLoader(val_loader,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=True)

    count = 0
    json_file_batch_path = 'bs{}_{}.json'.format(str(args.batch_size),args.json_path)
    target_dict = {}
    target_batch_dict = {}

    for i, (data, target, video_subpath) in enumerate(inference_loader):
        l = len(target)
        print('preprocessing ' + str(count + 1) + ' to ' + str(count + l))
        keys = list(range(count + 1, count + 1 + l))
        values = target.tolist()
        target_dict.update(dict(zip(keys, values)))
        target_batch_dict.update({str(count + 1) + '_' + str(count + l): values})

        batch_bin = data.cpu().numpy()
        if l == args.batch_size:
            batch_bin.tofile(str(save_path) + '/' + 'UCF101_batch_' + str(count + 1) + '_' + str(count + l) + '.bin')
            count = count + l
        else:
            print('remove this bin because not enough data in this bin')

    with open(json_file_batch_path, 'w') as f:
        json_str = json.dumps(target_batch_dict, indent=4, ensure_ascii=False)
        f.write(json_str)
