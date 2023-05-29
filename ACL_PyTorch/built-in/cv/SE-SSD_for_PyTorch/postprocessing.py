# Copyright 2023 Huawei Technologies Co., Ltd
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
import json
import argparse
 
import torch
import numpy as np
from torch.utils.data import DataLoader
 
from det3d import torchie
from det3d.models import build_detector
from det3d.models.detectors.voxelnet_sessd import VoxelNet
from det3d.datasets import build_dataset
from det3d.datasets.kitti.kitti import KittiDataset
from det3d.torchie.parallel import collate_kitti
from det3d.torchie.utils.config import ConfigDict
import middle_conv
 
 
def truncate_padding(preds_dict: dict, example: dict, batch_size: int) -> dict:
    '''
    Truncate the zero paddings.
    '''
    active_length = example["anchors"][0].shape[0]
    if batch_size != active_length:
        for k, v in preds_dict.items():
            unit_length = v.shape[0] // batch_size
            preds_dict[k] = v[:active_length * unit_length]
 
    return preds_dict
 
 
def post_processing(
        model: VoxelNet, 
        dataset: KittiDataset, 
        batch_size: int, 
        test_cfg: ConfigDict, 
        preds_dir: str
    ) -> dict:
    pred_file_list = sorted(os.listdir(preds_dir))
    pred_file_list = [os.path.join(preds_dir, file_name) for file_name in pred_file_list]
    pred_file_list = [pred_file_list[i:i+4] for i in range(0,len(pred_file_list),4)]
    num_preds = len(pred_file_list)
 
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=None, collate_fn=collate_kitti, shuffle=False,)
 
    results_dict = {}
    for task_id, (example, (box_file, cls_file, dir_cls_file, iou_file)) in enumerate(zip(data_loader, pred_file_list)):
        print(f"[{task_id + 1} / {num_preds}]", end="\r")
 
        box_preds = torch.Tensor(np.fromfile(box_file, dtype=np.float32))
        cls_preds = torch.Tensor(np.fromfile(cls_file, dtype=np.float32))
        dir_cls_preds = torch.Tensor(np.fromfile(dir_cls_file, dtype=np.float32))
        iou_preds = torch.Tensor(np.fromfile(iou_file, dtype=np.float32))
 
        preds_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
            "dir_cls_preds": dir_cls_preds,
            "iou_preds": iou_preds,
        }
 
        preds_dict = truncate_padding(preds_dict, example, batch_size)
 
        predict_results = model.bbox_head.predict(example, [preds_dict], test_cfg)
        for predict_result in predict_results:
            token = predict_result["metadata"]["token"]
            results_dict.update({token: predict_result})
 
    return results_dict
 
 
def evaluation(dataset: KittiDataset, _results: dict) -> None:
    result_dict, detections = dataset.evaluation(_results)
 
    for k, v in result_dict["results"].items():
        print(f"Evaluation {k}: {v}")
 
    for k, v in result_dict["results_2"].items():
        print(f"Evaluation {k}: {v}")
 
 
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default='./SE-SSD/examples/second/configs/config.py', 
        help="Config file path"
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root path of dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of data.")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Path of model output diractory.")
    parser.add_argument(
        "--info_file", 
        type=str, 
        default='./kitti_infos_val.pkl', 
        help="Path of the dataset information pkl file."
    )
    parser.add_argument("--save_dir", type=str, default='.', help="Path to save the post-processed files.")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    settings = parse_arguments()
 
    config = torchie.Config.fromfile(settings.config)
    
    detector = build_detector(config.model, train_cfg=None, test_cfg=config.test_cfg)
 
    # build dataset
    valset_config = config.data.val
    valset_config.test_mode = True
    valset_config.root_path = settings.data_root
    valset_config.info_path = settings.info_file
    val_dataset = build_dataset(valset_config)
 
    results = post_processing(detector, val_dataset, settings.batch_size, config.test_cfg, settings.model_output_dir)
 
    # Calculate average precision
    print("Calculating the average precision, please wait for a while...")
    evaluation(val_dataset, results)
    
    # convert torch.Tensor to list, convert PosixPath to string
    for key in results.keys():
        for (k_inner, v_inner) in results[key].items():
            if isinstance(v_inner, torch.Tensor):
                results[key][k_inner] = v_inner.tolist()
 
            elif isinstance(v_inner, dict):
                results[key][k_inner]["image_prefix"] = str(results[key][k_inner]["image_prefix"])
                results[key][k_inner]["image_shape"] = results[key][k_inner]["image_shape"].tolist()
 
    # save result to json
    if not os.path.exists(settings.save_dir):
        os.makedirs(settings.save_dir)
 
    save_json_path = os.path.join(settings.save_dir, "result.json")
    results_json = json.dumps(results)
 
 
    with os.fdopen(os.open(save_json_path, os.O_RDWR|os.O_CREAT, 0o644), "w") as outfile:
        outfile.write(results_json)
 
    print(f"The result has been saved to {save_json_path}")
    print("Done.")