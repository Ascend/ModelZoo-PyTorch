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
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json

from run.utils import AverageMeter


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


def test(data_loader, model, opt, class_names, logger, device_ids=0):
    if device_ids == 0:
        print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if opt.gpu_or_npu == 'npu' and not opt.no_drive:
            inputs = inputs.to(opt.device, non_blocking=True)

        elif opt.gpu_or_npu == 'gpu' and not opt.no_drive:
            inputs = inputs.to(opt.device)

        with torch.no_grad():
            inputs = Variable(inputs)

        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id, test_results, class_names)

                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]

        if (i % 100) == 0:
                with open(os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)), 'w') as f:
                    json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i % 50 == 0) or (i+1 == len(data_loader)):
            if device_ids == 0:  # distributed master or 1p
                print('[{}/{}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                          i + 1,
                          len(data_loader),
                          batch_time=batch_time,
                          data_time=data_time))

    # video acc
    if device_ids == 0:  # distributed master or 1p
        with open(os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)), 'w') as f:
            json.dump(test_results, f)

        from run.eval_ucf101 import UCFclassification
        ucf_classification = UCFclassification(opt.annotation_path,
                                               os.path.join(opt.root_path, opt.result_path+'/val.json'),
                                               subset='validation', top_k=1)
        ucf_classification.evaluate()

        print("test top1 acc: " + str(ucf_classification.hit_at_k))
        logger.log({'top1 acc': str(ucf_classification.hit_at_k)})
