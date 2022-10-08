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
import cv2
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import codecs
import shutil

from tqdm import tqdm
from datasets import UCAS_AODDataset
from utils.bbox import rbox_2_quad
from utils.utils import sort_corners, is_image
from utils.map import eval_mAP

from opts import parse_opts_eval
from models.main_model import RetinaNetNPU
from main_utils import static_im_detect


DATASETS = {"UCAS_AOD": UCAS_AODDataset}


def data_evaluate(model,
                  target_size,
                  test_path,
                  conf,
                  dataset,
                  root_dir):
    out_dir = os.path.join(root_dir, 'detection-results')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    ds = DATASETS[dataset]()

    with open(test_path, 'r') as f:
        if dataset == 'VOC':
            im_dir = test_path.replace('/ImageSets/Main/test.txt', '/JPEGImages')
            ims_list = [os.path.join(im_dir, x.strip('\n') + '.jpg') for x in f.readlines()]
        else:
            ims_list = [x.strip('\n') for x in f.readlines() if is_image(x.strip('\n'))]
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    if isinstance(target_size, int):
        target_size_h, target_size_w = target_size, target_size
    else:
        target_size_h, target_size_w = target_size
    for idx, im_path in enumerate(tqdm(ims_list, desc=s)):
        im_name = os.path.split(im_path)[1]
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = static_im_detect(model, im, target_size_h, target_size_w, conf=conf)
        nt += len(dets)
        out_file = os.path.join(out_dir, im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            res = sort_corners(rbox_2_quad(dets[:, 2:]))
            for k in range(dets.shape[0]):
                f.write('{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                    ds.return_class(dets[k, 0]), dets[k, 1],
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                )
        assert len(os.listdir(os.path.join(root_dir, 'ground-truth'))) != 0, 'No labels found in test/ground-truth!! '
    mAP = eval_mAP(root_dir, use_07_metric=False, if_draw=False)
    # display result
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0))
    print("result_mAP: ", mAP)
    return 0, 0, mAP, 0


def evaluate(target_size,
             test_path,
             dataset,
             root_dir,
             backbone=None,
             weight=None,
             model=None,
             num_classes=10,
             conf=0.3):
    if model is None:
        model = RetinaNetNPU(backbone=backbone, num_classes=num_classes)
        if weight.endswith('.pth'):
            chkpt = torch.load(weight)
            # load model
            if 'model' in chkpt.keys():
                model.load_state_dict(chkpt['model'])
            else:
                model.load_state_dict(chkpt)

    model.eval()
    model.npu()

    if dataset in ["UCAS_AOD"]:
        results = data_evaluate(model, target_size, test_path, conf, dataset, root_dir)
    else:
        raise RuntimeError('Unsupported dataset!')
    return results


def main():
    opt = parse_opts_eval()
    target_size = [int(size) for size in opt.target_size.split(",")]
    device_id = int(opt.device_index)
    opt.device = torch.device("npu:{}".format(device_id))
    torch.npu.set_device(opt.device)
    evaluate(target_size,
             opt.test_path,
             opt.dataset,
             opt.root_path,
             opt.backbone,
             opt.weight,
             num_classes=opt.num_classes)


if __name__ == '__main__':
    main()
