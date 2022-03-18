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
from dist_utils import is_main_process, dist_print, get_rank, get_world_size, synchronize
import os, json, torch, scipy
import numpy as np
from sklearn.linear_model import LinearRegression
import shutil

class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        # pred：pred中'lane'; gt：ground_truth中'lane'; y_sample： ground_truth中 'h_sample'; pred中'runtime'
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.), 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:  # 直接将pred_file里的文字变成json 放在list里
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]  # 把 ground_truth放进list里
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file'].replace('/', ''): l for l in json_gt}  # ground_truth的dict
        accuracy, fp, fn = 0., 0., 0.  # 初始化acc fp fn
        for pred in json_pred:  # 遍历 predict的json
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']  # pred中'lanes'项
            run_time = pred['run_time']
            if raw_file not in gts.keys():
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']  # ground_truth中'lanes' 项
            y_samples = gt['h_samples']
            try:  # 使用bench函数进行评价
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ])

def generate_tusimple_lines(out, griding_num=100, localization_type='rel'):
    out = out.data.cpu().numpy()
    # out = out.data.cpu().numpy().reshape(1,101,56,4)
    out_loc = np.argmax(out, axis=0)

    if localization_type == 'rel':
        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(griding_num)
        idx = idx.reshape(-1, 1, 1)

        loc = np.sum(prob * idx, axis=0)

        loc[out_loc == griding_num] = griding_num
        out_loc = loc
    lanes = []
    for i in range(out_loc.shape[1]):
        out_i = out_loc[:, i]
        lane = [int(round((loc + 0.5) * 1280.0 / (griding_num - 1))) if loc != griding_num else -2 for loc in out_i]
        lanes.append(lane)
    return lanes


def run_test_tusimple(work_dir, exp_name, griding_num=100, use_aux=True, distributed=False, batch_size=1):
    print('Start Processing Predict Lanes...')
    output_path = os.path.join(work_dir,exp_name+'.%d.txt' % get_rank())
    fp = open(output_path,'w')

    f = open(os.path.join(work_dir,'test.txt'),'r')
    image_path = f.read().splitlines()
    image_path_size = len(image_path)

    for i in range(image_path_size):
        name = image_path[i].replace('/','')
        relative_eval_path = image_path[i].replace('/','').replace('.jpg','_1.txt')
        inference_file = open(os.path.join(work_dir,relative_eval_path))
        inference_data = inference_file.readlines()[0].replace('\n','')
        inference_data_array = np.array([float(i) for i in inference_data.split(' ') if i != '']).reshape(1, griding_num+1,56,4)
        # inference_data_array 的纬度是（1,101,56,4)
        inference_data_tensor = torch.from_numpy(inference_data_array)

        # _shape = inference_data_array[0,0].shape
        # 网络输出在这里，是一个tensor，对OM模型的推理结果convert_inference_output.py的方法，np reshape之后转tensor
        out = inference_data_tensor
        _shape = inference_data_tensor.size()

        if len(out)==2 and use_aux:
          out = out[0]
        tmp_dict = {}
        tmp_dict['h_samples'] = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
                                 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
                                 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
                                 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
        tmp_dict['raw_file'] = name
        tmp_dict['run_time'] = 10
        tmp_dict['lanes'] = generate_tusimple_lines(out[0], griding_num = 100)

        json_str = json.dumps(tmp_dict)
        print('Predict Lane Processed: %d/2782' % (i+1))
        fp.write(json_str + '\n')
    fp.close()
    print('Predict File generated.')

def combine_tusimple_test(work_dir, exp_name):
    size = get_world_size()
    all_res = []
    for i in range(size):
        output_path = os.path.join(work_dir, exp_name+'.%d.txt' % i)
        with open(output_path, 'r') as fp:
            res = fp.readlines()
        all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        name = res[5:]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(work_dir,exp_name+'.txt')
    with open(output_path,'w') as fp:
        fp.writelines(all_res_no_dup)

def eval_lane(work_dir, griding_num=100, use_aux=True, distributed=False):
    exp_name = 'tusimple_eval_temp'
    run_test_tusimple(work_dir, exp_name, griding_num, use_aux, distributed)
    synchronize()    # 等待所有结果输出

    if is_main_process():
        combine_tusimple_test(work_dir,exp_name)
        res = LaneEval.bench_one_submit(os.path.join(work_dir,exp_name+'.txt'),os.path.join(work_dir,'test_label.json'))
        res = json.loads(res)
        for r in res:
            dist_print(r['name'], r['value'])
    synchronize()

if __name__ == '__main__':
    dir_path = sys.argv[1]
    dataset_path = sys.argv[2]
    type = np.float16

    src1 = dataset_path + '/test.txt'
    dst1 = dir_path + '/test.txt'
    shutil.copyfile(src1,dst1)

    src2 = dataset_path + '/test_label.json'
    dst2 = dir_path + '/test_label.json'
    shutil.copyfile(src2,dst2)
    print('Test Labels Transferred.')


    for file_path, _, file_names in os.walk(dir_path):
        for file_name in file_names:
            if ".bin" in file_name:
                src_abs_path = os.path.abspath(os.path.join(file_path, file_name))
                data = np.fromfile(src_abs_path, dtype=type)
                dst_abs_path = src_abs_path.replace('.bin', '.txt')
                fo = open(dst_abs_path, "w")
                for x in data:
                    s = str(x)
                    fo.write(s + " ")
                fo.close()
    print('txt file generated.')
    eval_lane(dir_path)

