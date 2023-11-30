# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import argparse
import time

import torch
import torch_aie
from tqdm import tqdm

from yolox.data import COCODataset, ValTransform
from yolox.evaluators import COCOEvaluator
from yolox.utils.boxes import postprocess
from yolox.utils.demo_utils import demo_postprocess


def main():
    print("[INFO] YoloX Torch-AIE evaluation process start")

    parser = argparse.ArgumentParser(description="YOLOX Preprocess")
    parser.add_argument('--dataroot', dest='dataroot',
                        help='data root dirname', default='/data/datasets/coco',
                        type=str)
    parser.add_argument('--batch',
                        help='validation batch size', default=1,
                        type=int)
    parser.add_argument('--ts',
                        help='root of ts module', default="./yoloxb1_torch_aie.pt",
                        type=str)
    parser.add_argument('--device-id', default=0, type=int, help='npu device id')
    opt = parser.parse_args()

    valdataset = COCODataset(
        data_dir=opt.dataroot,
        json_file='instances_val2017.json',
        name="val2017",
        img_size=(640, 640),
        preproc=ValTransform(legacy=False),
    )
    sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
        "num_workers": 8, "pin_memory": True, "sampler": sampler, "batch_size": opt.batch
    }

    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    torch_aie.set_device(opt.device_id)

    print("[INFO] Loading TS module")
    print(opt.ts)
    aie_module = torch.jit.load(opt.ts)
    aie_module.eval()

    inference_time = []
    data_list = []
    coco_evaluator = COCOEvaluator(val_loader, img_size=(640, 640), confthre=0.001, nmsthre=0.65, num_classes=80)

    print(f"[INFO] Start Torch-AIE inference, please be patient (batch={opt.batch})")
    for idx, datas in enumerate(tqdm(val_loader)):
        input_npu = datas[0].to(f"npu:{opt.device_id}")
        stream = torch_aie.npu.Stream(f"npu:{opt.device_id}")

        with torch_aie.npu.stream(stream):
            # Inference with Torch-AIE-compiled TS module
            start = time.time()
            output_npu = aie_module(input_npu)
            stream.synchronize()
            cost = time.time() - start
            # Warm-up using 5 steps
            if idx >= 5:
                inference_time.append(cost)

        output_cpu = output_npu.to('cpu')
        outputs = demo_postprocess(output_cpu, [640, 640])
        outputs = postprocess(outputs, num_classes=80, conf_thre=0.001, nms_thre=0.65)
        data_list.extend(coco_evaluator.convert_to_coco_format(outputs, datas[2], datas[3]))

    # Use COCO_Evaluator to evaluate the accuracy
    if idx < 5:
        print("\n[WARNING] The number of samples is too small, please use more samples for evaluation")
    else:
        coco_result = coco_evaluator.evaluate_prediction(data_list)
        print(coco_result)
        print(f'\n[INFO] Torch-AIE inference avg cost: {sum(inference_time) / len(inference_time) * 1000 / opt.batch} ms/pic')
        print(f'[INFO] Throughput = {opt.batch / (sum(inference_time) / len(inference_time))} pic/s')

    print('[INFO] YoloX Torch-AIE evaluation process finished')

if __name__ == "__main__":
    main()
