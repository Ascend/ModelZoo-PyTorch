# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment

import collections
import os
import shutil
import random
import argparse
import time
import glob

from datetime import datetime
from tqdm import tqdm

import monai
import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split

from segment_anything import sam_model_registry

join = os.path.join

# set seeds
torch.manual_seed(2023)
torch.npu.empty_cache()


def init_env():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tr_npy_path",
        type=str,
        default="data/npy/CT_Abd",
        help="path to training npy files; two subfolders: gts and imgs",
    )
    parser.add_argument(
        "--task_name", type=str, default="MedSAM-ViT-B"
    )
    parser.add_argument(
        "--model_type", type=str, default="vit_b"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="work_dir/MedSAM/sam_vit_b_01ec64.pth"
    )

    parser.add_argument(
        "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default=""
    )
    parser.add_argument(
        "--work_dir", type=str, default="./work_dir"
    )

    # train
    parser.add_argument(
        "--num_epochs", type=int, default=1000
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--num_workers", type=int, default=8
    )

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
    )
    parser.add_argument(
        "--use_wandb", type=bool, default=False, help="use wandb to monitor training"
    )
    parser.add_argument(
        "--use_amp", action="store_true", default=False, help="use amp"
    )
    parser.add_argument(
        "--bucket_cap_mb",
        type=int,
        default=25,
        help="The amount of memory in Mb that DDP will accumulate before "
             "firing off gradient communication for the bucket (need to tune)",
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps before syncing gradients for backprop",
    )
    parser.add_argument(
        "--resume", type=str, default="", help="Resuming training from checkpoint"
    )

    # Distributed relative
    parser.add_argument('--init_method', default='env://', help='env or tcp')
    parser.add_argument('--node_rank', type=int, default=0, help='node id')
    parser.add_argument('--nnodes', type=int, default=1, help='total nodes')
    parser.add_argument('--nproc_per_node', type=int, default=torch.npu.device_count(), help='nproc_per_node')

    args = parser.parse_args()
    args.world_size = args.nnodes * args.nproc_per_node

    return args


def init_wandb(args):
    if args.use_wandb:
        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.tr_npy_path,
                "model_type": args.model_type,
            },
        )


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
        # do not compute gradients for prompt encoder
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
        return ori_res_masks


def main(args):
    mp.spawn(main_worker, nprocs=args.nproc_per_node, args=(args.nproc_per_node, args))


def main_worker(local_rank, nprocs, args):
    rank = local_rank + args.node_rank * args.nproc_per_node
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)

    print(f"[Rank {rank}]: Use NPU: {local_rank} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.npu.set_device(local_rank)
    torch.distributed.init_process_group(
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
        find_unused_parameters=True
    )

    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )

    # Setting up optimiser and loss func
    img_mask_encdec_params = list(
        medsam_model.module.image_encoder.parameters()
    ) + list(medsam_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    dataset_total = NpyDataset(args.tr_npy_path)
    train_set, val_set = random_split(dataset_total, [0.8, 0.2])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    print("Number of training samples: ", len(train_set))
    print("Number of evaluation samples: ", len(val_set))
    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    eval_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=eval_sampler,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            loc = "npu:{}".format(local_rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                rank,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()

    if args.use_amp:
        scaler = torch.npu.amp.GradScaler()
        print(f"[RANK {rank}: NPU {local_rank}] Using AMP for training")

    step = 1
    total_steps = 0
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        for step, (image, gt_2d, boxes, _) in enumerate(
                tqdm(train_dataloader, desc=f"[Epoch: {epoch + 1}/{num_epochs}][RANK {rank}: NPU {local_rank}]")
        ):
            total_steps += 1
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()

            image, gt_2d = image.npu(), gt_2d.npu()
            if args.use_amp:
                with torch.autocast(device_type="npu", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt_2d) + ce_loss(
                        medsam_pred.float(), gt_2d.float()
                    )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt_2d) + ce_loss(
                    medsam_pred.float(), gt_2d.float()
                )

                # Gradient accumulation
                if args.grad_acc_steps > 1:
                    loss = (
                            loss / args.grad_acc_steps
                    )  # normalize the loss because it is accumulated
                    if (step + 1) % args.grad_acc_steps == 0:
                        ## Perform gradient sync
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        ## Accumulate gradient on current node without backproping
                        with medsam_model.no_sync():
                            ## calculate the gradient only
                            loss.backward()  
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # If you want to see the loss in every step and every rank.
            # print(f'[{time.asctime()}][RANK-{rank}][EPOCH-{epoch}][STEP-{total_steps}] '
            #       f'Loss: {loss.detach().cpu().float()}')

            if step > 10 and step % 100 == 0:
                if is_main_host:
                    checkpoint = {
                        "model": medsam_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(
                        checkpoint,
                        join(model_save_path, "medsam_model_latest_step.pth"),
                    )

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})

        print(f'[{time.asctime()}][RANK-{rank}][EPOCH-{epoch}] Loss: {epoch_loss}')

        # save the model checkpoint
        if is_main_host:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

            # save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
        torch.distributed.barrier()

        # plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #comment this line if you are running on a server: plt.show()
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    arguments = init_env()
    init_wandb(arguments)
    main(arguments)
