import os
import argparse
import numpy as np
import torch
from mmdet.core import bbox2result, encode_mask_results
from mmdet.datasets import CocoDataset
from torchvision.models.detection.roi_heads import paste_masks_in_image
from tqdm import tqdm
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask


def postprecess():
    dataset = CocoDataset(ann_file=args.ann_file_path, pipeline=[])
    bin_path = args.bin_file_path
    latest_result = os.listdir(bin_path)
    latest_result.sort()
    bin_path = os.path.join(bin_path, latest_result[-1])
    model_h = args.input_height
    model_w = args.input_width

    results = []
    for data_info in tqdm(dataset.data_infos):
        file_name = data_info['file_name']
        ori_h = data_info['height']
        ori_w = data_info['width']
        scalar_ratio = min(model_h / ori_h, model_w / ori_w)

        path_base = os.path.join(bin_path, file_name.split('.')[0])
        bboxes = np.fromfile(path_base + '_output_0.bin', dtype=np.float32)
        bboxes = np.reshape(bboxes, [100, 5])
        labels = np.fromfile(path_base + '_output_1.bin', dtype=np.int64)
        mask_pred = np.fromfile(path_base + '_output_2.bin', dtype=np.float32)
        mask_pred = np.reshape(mask_pred, [100, 1, 28, 28])

        bboxes[..., 0:4] = bboxes[..., 0:4] / scalar_ratio

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        mask_pred = torch.from_numpy(mask_pred).to(device)

        bboxes_tensor = torch.from_numpy(bboxes[..., 0:4]).to(device)
        img_h = ori_h
        img_w = ori_w
        N = 100
        threshold = 0.5
        
        masks, spatial_inds = _do_paste_mask(
            mask_pred,
            bboxes_tensor,
            img_h,
            img_w,
            skip_empty=device.type == 'cpu')
        masks = (masks >= threshold).to(dtype=torch.bool)

        im_mask = torch.zeros(N, img_h, img_w, device=device, dtype=torch.bool)
        im_mask[(torch.arange(N), ) + spatial_inds] = masks
        segms = im_mask.squeeze(1).cpu().numpy()

        cls_segms = [[] for _ in range(80)]
        for label, segm in zip(labels, segms):
            cls_segms[label].append(segm)
        cls_bboxes = bbox2result(bboxes, labels, 80)
        results.append((cls_bboxes, encode_mask_results(cls_segms)))
    
    dataset.evaluate(results, metric=['bbox', 'segm'], classwise=True, jsonfile_prefix='./')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file_path", default="./data/coco/annotations/instances_val2017.json")
    parser.add_argument("--bin_file_path", default="./result/")
    parser.add_argument("--input_height", default=800, type=int)
    parser.add_argument("--input_width", default=1344, type=int)
    args = parser.parse_args()

    postprecess()