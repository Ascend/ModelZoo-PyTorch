# Copyright 2021 Huawei Technologies Co., Ltd

import json
import sys

f = open(sys.argv[1], 'r')
gpu_nums = int(sys.argv[2])
batch_size = int(sys.argv[3])

epoch = 0
cnt = 0
fps = 0
total_fps = 0
for data in f:
    line = json.loads(data)
    if not 'mode' in line:
        pass
    elif line['mode'] == 'train':
        # ignore first 50 iters
        if line['iter'] > 50:
            cnt = cnt + 1
            fps = fps + gpu_nums * batch_size / line['time']
    elif line['mode'] == 'val':
        epoch = epoch + 1
        print({'epoch': epoch, 'fps': fps / cnt})
        total_fps = total_fps + fps / cnt
        cnt = 0
        fps = 0
print({'fps': total_fps / epoch})
