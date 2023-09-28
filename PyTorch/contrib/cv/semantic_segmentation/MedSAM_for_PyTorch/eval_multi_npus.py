# -*- coding: utf-8 -*-
"""
eval the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment

import collections
import os
import shutil
import random
import argparse
import glob

from datetime import datetime
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split

from utils import SurfaceDice
from segment_anything import sam_model_registry

join = os.path.join

# set seeds
torch.manual_seed(2023)
torch.npu.empty_cache()


def init_env():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/npy/CT_Abd",
        help="path to npy files; two sub-folders: gts and imgs",
    )
    parser.add_argument(
        "--task_name", type=str, default="MedSAM-ViT-B"
    )
    parser.add_argument(
        "--model_type", type=str, default="vit_b"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
    )

    parser.add_argument(
        "--work_dir", type=str, default="./work_dir"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8
    )

    # Distributed relative
    parser.add_argument('--init_method', default='env://', help='env or tcp')
    parser.add_argument('--node_rank', type=int, default=0, help='node id')
    parser.add_argument('--nnodes', type=int, default=1, help='total nodes')
    parser.add_argument('--nproc_per_node', type=int, default=torch.npu.device_count(), help='nproc_per_node')

    args = parser.parse_args()
    args.world_size = args.nnodes * args.nproc_per_node

    return args


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def compute_miou(pred, target, n_class):
    mini = 1

    # 计算公共区域
    intersection = pred * (pred == target)

    # 直方图
    area_inter, _ = np.histogram(intersection, bins=2, range=(mini, n_class))
    area_pred, _ = np.histogram(pred, bins=2, range=(mini, n_class))
    area_target, _ = np.histogram(target, bins=2, range=(mini, n_class))
    area_union = area_pred + area_target - area_inter

    # 交集已经小于并集
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    rate = round(max(area_inter) / max(area_union), 4)
    return rate


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
                np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
                "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt_2d = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)

        assert np.max(gt_2d) == 1 and np.min(gt_2d) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt_2d > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        h, w = gt_2d.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(w, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(h, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        Item = collections.namedtuple("Item", ["img_1024", "gt_2d", "bboxes", "img_name"])
        item = Item(
            torch.tensor(img_1024).float(), 
            torch.tensor(gt_2d[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name
        )
        return tuple(item)


class MedSAM(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        ori_res_masks = torch.sigmoid(ori_res_masks)
        ori_res_masks = (ori_res_masks > 0.7).type(torch.int8)

        return ori_res_masks


def main(args):
    mp.spawn(main_worker, nprocs=args.nproc_per_node, args=(args.nproc_per_node, args))


def main_worker(local_rank, nprocs, args):
    rank = local_rank + args.node_rank * args.nproc_per_node
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)

    print(f"[Rank {rank}]: Use NPU: {local_rank} for evaluating")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.npu.set_device(local_rank)

    dist.init_process_group(
        'hccl',
        init_method=args.init_method,
        world_size=args.world_size,
        rank=rank
    )

    sam_model = sam_model_registry.get(args.model_type)(checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    ).npu()

    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    medsam_model.eval()

    dsc = SurfaceDice.compute_dice_coefficient

    dataset_total = NpyDataset(args.data_path)
    train_set, val_set = random_split(dataset_total, [0.8, 0.2])

    eval_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    print("Number of evaluation samples: ", len(val_set))

    eval_dataloader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=eval_sampler,
    )

    total_steps = 0
    dsc_records = torch.Tensor([0.0])
    miou_records = torch.Tensor([0.0])
    for step, (image, gt_2d, boxes, _) in enumerate(tqdm(eval_dataloader)):
        total_steps += 1

        boxes_np = boxes.detach().cpu().numpy()
        image, gt_2d = image.npu(), gt_2d.npu()

        with torch.no_grad():
            medsam_pred = medsam_model(image, boxes_np)
            pred, gt = medsam_pred[0].cpu().numpy(), gt_2d[0].cpu().numpy()
            dsc_records += dsc(gt, pred)
            miou_records += compute_miou(pred, gt, n_class=2)

        if step == 0:
            # compare in images
            image = image[0, 2].cpu()
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))

            ax[0].imshow(image, cmap="gray")
            ax[1].imshow(image, cmap="gray")
            ax[0].axis('off')
            ax[1].axis('off')

            show_mask(gt_2d[0, 0].cpu().numpy(), ax[0], random_color=False)
            show_mask(medsam_pred[0].cpu().numpy(), ax[1], random_color=False)
            show_box(boxes[0].cpu().numpy(), ax[0])
            show_box(boxes[0].cpu().numpy(), ax[1])

            plt.tight_layout()
            plt.savefig(f"compare_{step}.jpg")

    print(f"DSC:{float(dsc_records / total_steps)}")
    print(f"MIOU:{float(miou_records / total_steps)}")


if __name__ == "__main__":
    arguments = init_env()
    main(arguments)
