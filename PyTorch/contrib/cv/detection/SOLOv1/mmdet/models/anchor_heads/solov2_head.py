# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
# from mmdet.ops import DeformConv, roi_align
from mmdet.core import multi_apply, matrix_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
INF = 1e8

def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

@HEADS.register_module
class SOLOv2Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,
                 ins_out_channels=64,
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_dcn_in_tower=False,
                 type_dcn=None):
        super(SOLOv2Head, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn_in_tower = use_dcn_in_tower
        self.type_dcn = type_dcn
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if self.use_dcn_in_tower:
                cfg_conv = dict(type=self.type_dcn)
            else:
                cfg_conv = self.conv_cfg

            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats,
                                                       list(range(len(self.seg_num_grids))),
                                                       eval=eval, upsampled_size=upsampled_size)
        return cate_pred, kernel_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device, dtype=torch.float16)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device, dtype=torch.float16)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
        
        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')

        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred

    def loss(self,
             cate_preds,  # list[5 x tensor(bs, 80, num_grids, num_grids)]
             kernel_preds,  # list[5 x tensor(bs, c, num_grids, num_grids)]
             ins_pred,  # tensor(bs, c, h, w)
             gt_bbox_list,  # list[bs x tensor(n, 4)]
             gt_label_list,  # list[bs x tensor(n,)]
             gt_mask_list,  # list[bs x tensor(n, 1344, 1344)]
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        '''
            ins_label_list:[img_num x [tensor(n, mask_h, mask_w) x 5]]
            cate_label_list:[img_num x [tensor(num_gird, num_gird) x 5]]
            ins_ind_label_list:[img_num x [tensor(num_grid **2, ) x 5]]
            grid_order_list:[img_num x [[] x 5]]
            pos_inds_mask_list [ img_num x [tensor(max_len * 9,n ) x 5]]'''
        # diff
        # change 每个图片中gt的数目不固定，这里固定gt到64
        MAX_LEN = 90
        # print('gt_bboxes_list[0].shape', gt_bbox_list[0].shape)
        # print('gt_mask[0].shape', gt_mask_list[0].shape, gt_mask_list[0].dtype)
        device = gt_bbox_list[0].device
        for idx, i in enumerate(gt_bbox_list):
            num_gt = i.shape[0]
            gt_bboxes_zero = torch.zeros(MAX_LEN, 4, dtype=torch.float32, device=device)
            maxindex = min(num_gt, MAX_LEN)
            gt_bboxes_zero[:maxindex] = gt_bbox_list[idx][:maxindex]
            gt_bbox_list[idx] = gt_bboxes_zero

            gt_label_list[idx] = gt_label_list[idx].int()
            # gt_labels_zero = torch.zeros(MAX_LEN).int().cuda()  # 1
            gt_labels_zero = torch.zeros(MAX_LEN, dtype=torch.int, device=device)
            gt_labels_zero[:maxindex] = gt_label_list[idx][:maxindex]
            gt_label_list[idx] = gt_labels_zero

            gt_mask_list[idx] = torch.from_numpy(gt_mask_list[idx]).to(device)
            gt_masks_zero = torch.zeros((MAX_LEN, 1344, 1344), dtype=torch.uint8, device=device)  # 1
            gt_masks_zero[:maxindex] = gt_mask_list[idx][:maxindex]
            gt_mask_list[idx] = gt_masks_zero

        # print('gt_bboxes_list[0].shape', gt_bbox_list[0].shape)
        # print('gt_mask[0].shape', gt_mask_list[0].shape, type(gt_mask_list[0]))

        mask_feat_size = ins_pred.size()[-2:]  # 1/4
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, pos_inds_mask_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size,
            MAX_LEN=MAX_LEN)

        # print('my grid_order_list', grid_order_list[0][0][:20])
        # ins  dynamic
        # [tensor(n1 + n2, mask_gt_h, mask_gt_w) x 5]
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]
        pos_masks = [torch.cat([pos_mask_level_img for pos_mask_level_img in pos_mask_level], 0)
                     for pos_mask_level in zip(*pos_inds_mask_list)]
        # [[tensor(c, n) x img_num] x 5]  tensor 为每个level下，每个img提取的卷积核  dynamic
        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        # print('my kernel_preds per level', kernel_preds[0][0][:5, 0])
        # generate masks  dynamic
        ins_pred = ins_pred
        ins_pred_list = []  # [tensor(n1+n2, msk_pre_h, msk_pre_w) x 5]
        for b_kernel_pred in kernel_preds:  # level
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):  # img

                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]  # this img‘s pred
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape  # N 为channel，1 * 1的卷积 N应该等于 pred的 c，I为核的数目
                cur_ins_pred = cur_ins_pred.unsqueeze(0)  # [1, c, msk_pre_h, msk_pre_w]     c = N
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)  # 整理成卷积核，(n, c, filter_h, fw)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)  # (n, msk_pre_h, msk_pre_w)
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)  # 将所有img的同一level下的mask 合并，(n1+n2, msk_pre_h, msk_pre_w)
            ins_pred_list.append(b_mask_pred)
        # print('my ins_pred_list', ins_pred_list[0][0])
        ins_ind_labels = [  # [tensor(num_gird ** 2 x img, ) x 5]
            torch.cat([ins_ind_labels_level_img  # .flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)  # tensor num_gird ** 2 x img x 5

        num_ins = flatten_ins_ind_labels.sum()
        # print('num_ins', num_ins)
        # print('pos inds num', sum(pos_inds_list[0]) + sum(pos_inds_list[1]))
        # dice loss dynamic
        loss_ins = []
        # ins_pred_list [tensor(n1+n2, msk_pre_h, msk_pre_w) x 5]
        # ins_labels [tensor(n1+n2, mask_gt_h, mask_gt_w) x 5]
        # pos_masks  [tensor(n1+n2,) x 5]
        avg_factor = 1e-6
        for input, target, mask in zip(ins_pred_list, ins_labels, pos_masks):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target) * mask)  # tensor(n1 + n2, )
            avg_factor += torch.sum(mask > 0).float().item()
        # loss_ins = torch.cat(loss_ins).mean()  # tensor(5*(n1 + n2), )
        loss_ins = torch.cat(loss_ins).sum() / avg_factor

        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [  # [tensor(num_gird ** 2 x img, ) x 5]
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)  # tensor(num_gird ** 2 x img x 5, )

        cate_preds = [  # [tensor(num_gird ** 2 x img, 80) x 5]
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)  # tensor(num_gird ** 2 x img x 5, 80)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        # print('loss_cate', loss_cate, 'loss_ins', loss_ins)
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_cate)

    def solov2_target_single(self,  # tensor   gt in per img
                             gt_bboxes_raw,  # tensor(n, 4)
                             gt_labels_raw,  # tensor(n, )
                             gt_masks_raw,  # tensor(n, 1344, 1344)   固定后这里n = 64
                             mask_feat_size,
                             MAX_LEN=64):
        # print('gt_bboxes_raw', gt_bboxes_raw.shape)
        # print('gt_labels_raw', gt_labels_raw.shape)
        # print('gt_masks_raw', gt_masks_raw.shape)
        # process pre img target
        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))  # （n,)

        ins_label_list = []  # [tensor(n, h, w) x 5]                 ins_label_list中tensor第0维大小不同，原因是每个特征尺度下选中的位置不同
        cate_label_list = []  # [tensor(num_grid, num_gird) x 5]      每个特征尺度下添加 固定大小的tensor 不会动态
        ins_ind_label_list = []  # [tensor(num_grid * *2, )]             每个特征尺度下添加 固定大小的tensor 不会动态, 用于计算分类的 pos inds数目
        grid_order_list = []  # [[] x 5]            grid_order_list中每个列表的长度不同，列表长度取决于选中位置数目，引起后面loss部分卷积核大小不同
        pos_inds_mask_list = []
        # 一个level一个level的处理
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):
            # dynamic shape  nonzero算子提取满足条件索引，动态
            # hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()  # (n,)
            hit_masks = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound))
            # print('hit_masks', hit_masks.shape)
            # print('hit_indices', hit_indices.shape)
            # num_ins = len(hit_indices)

            # ins_label = []   # 存放每张图每个level下的 valid mask，最后会cat成一个tensor，所以会随着选中位置数目不同最后形成动态shape的tensor
            # grid_order = []  # 序号
            grid_order = torch.zeros([MAX_LEN * 9], dtype=torch.int64, device=device)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool,
                                        device=device)  # bool类型，记录哪些位置被选中，loss部分用到了，用来计算选中位置的数目，计算分类损失
            ins_label = torch.zeros([MAX_LEN * 9, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                    device=device)  # 实例mask
            pos_inds_mask = torch.zeros([MAX_LEN * 9], dtype=torch.bool, device=device)
            pos_inds_num = 0
            if hit_masks.sum() == 0:
                # ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append(grid_order)
                pos_inds_mask_list.append(pos_inds_mask)
                continue
            # gt_bboxes = gt_bboxes_raw[hit_indices]  # tensor(n, 4)    # 根据nonzero算子提取的索引进行提取，动态的
            # gt_labels = gt_labels_raw[hit_indices]  # tensor(n,)
            # gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]  # numpy(n, h, w)

            gt_bboxes = gt_bboxes_raw * hit_masks.view(-1, 1)
            gt_labels = gt_labels_raw * hit_masks
            # gt_masks = gt_masks_raw * hit_masks.view(-1, 1, 1).cpu().numpy()
            gt_masks = gt_masks_raw * hit_masks.view(-1, 1, 1).expand_as(gt_masks_raw)

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # half_ws1 = 0.5 * (gt_bboxes1[:, 2] - gt_bboxes1[:, 0]) * self.sigma
            # half_hs1 = 0.5 * (gt_bboxes1[:, 3] - gt_bboxes1[:, 1]) * self.sigma

            # mass center
            # gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)  # tensor(n, ori_h, ori_w)
            # center_ws, center_hs = center_of_mass(gt_masks_pt)

            center_ws, center_hs = center_of_mass(gt_masks)

            # valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0  # 这里认为即使面积符合范围，也可能有空的mask？ 这里我们尝试用
            output_stride = 4
            # according valid gt's center,hs.., perpare target
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels,
                                                                                               half_hs, half_ws,
                                                                                               center_hs, center_ws,
                                                                                               valid_mask_flags):
                if not valid_mask_flag:
                    # print('gt is zero')
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)
                # cate_label[top:(down + 1), left:(right + 1)] = gt_label
                seg_mask = mmcv.imrescale(seg_mask.cpu().numpy(), scale=1. / output_stride)  # seg_mask 原图大小的，这里下采样
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        cate_label[i, j] = gt_label
                        label = int(i * num_grid + j)
                        # cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                        #                             device=device)
                        # cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        # ins_label.append(cur_ins_label)
                        ins_label[pos_inds_num] = seg_mask  # v2中 同一个位置被不同实例选中，这个位置要负责预测两个mask，所以不能覆盖，v1是后者覆盖前者
                        ins_ind_label[label] = True  # 如果label位置被多次选中，只记为1次做分类的avg_factor
                        # grid_order.append(label)
                        grid_order[pos_inds_num] = label
                        pos_inds_mask[pos_inds_num] = True
                        pos_inds_num += 1

            ins_label_list.append(ins_label)  # [tensor(n, mask_feat_size, mask_feat_size) x 5]
            cate_label_list.append(cate_label)  # [tensor(num_grid * num_grid) x 5]
            ins_ind_label_list.append(ins_ind_label)  # [tensor(num_grid **2,) x 5]
            grid_order_list.append(grid_order)  # [[] x 5]
            pos_inds_mask_list.append(pos_inds_mask)  # [5]
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, pos_inds_mask_list

    def get_seg(self, cate_preds, kernel_preds, seg_pred, img_metas, cfg, rescale=None):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)
            result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       kernel_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):

        assert len(cate_preds) == len(kernel_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]
        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        # mask.
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                    kernel=cfg.kernel,sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores
