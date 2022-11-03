# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import cv2

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
def visualize_lambda(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, epoch=-1, print_prefix=''):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape

            lambda_vals = [0, 1]

            se_activation_output_dict = {}

            for lambda_idx, lambda_val in enumerate(lambda_vals):
                lambda_a = lambda_val*torch.ones(B, 1).cuda()
                lambda_vec = torch.cat([lambda_a, 1 - lambda_a], dim=1)        
                output, se_list, embedding_list, before_x_list, after_x_list = model(input, lambda_vec)

                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

                if (i % config.PRINT_FREQ == 0) or (i == (len(val_loader)-1)):
                    save_size = 16
                    
                    meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])
                    
                    prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)
                    suffix = 'a'
                    for count in range(min(save_size, len(lambda_a))):
                        suffix += '_[{}:{}]'.format(count, round(lambda_a[count].item(), 2))

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                                      prefix, suffix)

                    # max_len = 256
                    max_len = 384

                    # for k in range(min(save_size, B)):
                    #     embedding_image = np.zeros((len(embedding_list), max_len)) ## number of SE modules x max_len

                    #     for count in range(len(embedding_list)):
                    #         embedding = embedding_list[count][k] ## size of activation
                    #         embedding = embedding.detach().cpu().numpy()

                    #         se_activation = se_list[count][k]
                    #         se_activation = se_activation.detach().cpu().numpy()

                    #         x = x_list[count][k]
                    #         x = x.detach().cpu().numpy()

                    #         embedding_image[count, :len(embedding)] = embedding[:]*se_activation[:] 
                    #         ##embedding is lambda embedding, se_activation is from se. both are multiplied to input tensor

                    #         print()
                    #         # print('image:{}, module:{}'.format(k, count), se_activation)
                    #         # print('image:{}, module:{}'.format(k, count), embedding*se_activation)
                    #         # print('image:{}, module:{}'.format(k, count), se_activation)
                    #         print('image:{}, module:{}'.format(k, count), x)

                    #     image_name = '{}_embedding_[image{}:l{}].jpg'.format(prefix, k, lambda_val)

                    #     sns_plot = sns.heatmap(embedding_image, vmin=0, vmax=1)
                    #     figure = sns_plot.get_figure()
                    #     figure.savefig(image_name)
                    #     plt.close()

                    for k in range(min(save_size, B)):
                        embedding_image = np.zeros((len(embedding_list), max_len)) ## number of SE modules x max_len

                        # count = 0
                        # count = 1
                        # count = 2
                        # count = 3
                        count = 4
                        # count = len(embedding_list)-1

                        embedding = embedding_list[count][k] ## size of activation
                        embedding = embedding.detach().cpu().numpy()

                        se_activation = se_list[count][k]
                        se_activation = se_activation.detach().cpu().numpy()

                        ## keep it as torch tensor
                        before_x = before_x_list[count][k]
                        before_x = before_x.detach()

                        after_x = after_x_list[count][k]
                        after_x = after_x.detach()

                        input_sample = input[k].detach()

                        # print()
                        # print('image:{}, module:{}'.format(k, count), se_activation)

                        if k not in se_activation_output_dict.keys():
                            se_activation_output_dict[k] = {lambda_val: {'se': se_activation, 'embedding': embedding, 'before_x': before_x, 'after_x': after_x, 'input': input_sample}}

                        else:
                            se_activation_output_dict[k][lambda_val] = {'se': se_activation, 'embedding': embedding, 'before_x': before_x, 'after_x': after_x, 'input': input_sample}

            ## done for batch, with both l=0 and l=1
            for image_num in se_activation_output_dict.keys():
                mode0_activation = se_activation_output_dict[image_num][0]['se']
                mode1_activation = se_activation_output_dict[image_num][1]['se']

                mode0_embedding = se_activation_output_dict[image_num][0]['embedding']
                mode1_embedding = se_activation_output_dict[image_num][1]['embedding']

                mode0_before_x = se_activation_output_dict[image_num][0]['before_x']
                mode1_before_x = se_activation_output_dict[image_num][1]['before_x']
                diff_before_x = 100*(mode1_before_x - mode0_before_x)**2
                print(diff_before_x.sum())

                mode0_after_x = se_activation_output_dict[image_num][0]['after_x']
                mode1_after_x = se_activation_output_dict[image_num][1]['after_x']
                diff_after_x = 100*(mode1_after_x - mode0_after_x)**2
                print(diff_after_x.sum())

                mode0_input = se_activation_output_dict[image_num][0]['input']
                mode1_input = se_activation_output_dict[image_num][1]['input']

                diff_activation = mode1_activation - mode0_activation
                diff_embedding = mode1_embedding - mode0_embedding

                # print('image:{}, l:{}'.format(image_num, 0), lambda_mode0_activation)
                # print('image:{}, l:{}'.format(image_num, 1), lambda_mode1_activation)
                # print('image:{}, diff'.format(image_num), diff)
                # print('image:{}, sum:{}'.format(image_num, np.abs(diff).sum()))
                # print()

                prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)

                # ------------------------------------------
                image_name = '{}_image{}_before_l0.jpg'.format(prefix, image_num)
                mode0_before_image = get_activation_image(mode0_before_x, mode0_input, image_name)

                image_name = '{}_image{}_before_l1.jpg'.format(prefix, image_num)
                mode1_before_image = get_activation_image(mode1_before_x, mode1_input, image_name)

                diff_before_image = get_activation_image(diff_before_x, mode1_input, image_name)

                # ------------------------------------------
                image_name = '{}_image{}_after_l0.jpg'.format(prefix, image_num)
                mode0_after_image = get_activation_image(mode0_after_x, mode0_input, image_name)

                image_name = '{}_image{}_after_l1.jpg'.format(prefix, image_num)
                mode1_after_image = get_activation_image(mode1_after_x, mode1_input, image_name)

                diff_after_image = get_activation_image(diff_after_x, mode1_input, image_name)

                # ------------------------------------------
                image_name = '{}_image{}_before_both.jpg'.format(prefix, image_num)
                blank_image = np.zeros((mode0_before_image.shape[0], 10, 3))
                both_before_image = np.concatenate((mode0_before_image, blank_image, mode1_before_image, blank_image, diff_before_image), axis=1)
                cv2.imwrite(image_name, both_before_image)

                # ------------------------------------------
                image_name = '{}_image{}_after_both.jpg'.format(prefix, image_num)
                both_after_image = np.concatenate((mode0_after_image, blank_image, mode1_after_image, blank_image, diff_after_image), axis=1)
                cv2.imwrite(image_name, both_after_image)

                # ------------------------------------------
                plt.plot(range(len(mode0_activation)), mode0_activation, 'b-', label='l0')
                plt.plot(range(len(mode1_activation)), mode1_activation, 'r-', label='l1')
                plt.plot(range(len(diff_activation)), diff_activation, 'g-', label='diff')

                plt.legend(loc='best')

                image_name = '{}_image{}_se.jpg'.format(prefix, image_num)
                plt.savefig(image_name)
                plt.close()

                # ------------------------------------------
                plt.plot(range(len(mode0_embedding)), mode0_embedding, 'b-', label='l0')
                plt.plot(range(len(mode1_embedding)), mode1_embedding, 'r-', label='l1')
                plt.plot(range(len(mode1_embedding)), diff_embedding, 'g-', label='diff')
                plt.legend(loc='best')

                image_name = '{}_image{}_embedding.jpg'.format(prefix, image_num)
                plt.savefig(image_name)
                plt.close()


    return 
# --------------------------------------------------------------------------------
## activation: 48 x 96 x 72: C x H x W
## input: 3 x 384 x 388: 3 x high_res H x high_res W
def get_activation_image(activations, input, file_name):
    num_channels = activations.shape[0]
    heatmap_height = 64
    heatmap_width = 32

    image = input.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

    heatmaps = activations.mul(255).clamp(0, 255).byte().cpu().numpy()

    ## arrange as 6 x 8
    grid_image = np.zeros((6*heatmap_height, (num_channels//6)*heatmap_width, 3), dtype=np.uint8)

    for i in range(num_channels):
        row = i%6
        col = i//6

        heatmap = heatmaps[i, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        colored_heatmap = cv2.resize(colored_heatmap, (int(heatmap_width), int(heatmap_height)))
        masked_image = colored_heatmap*0.7 + resized_image*0.3

        height_begin = heatmap_height * row
        height_end = heatmap_height * (row + 1)

        width_begin = heatmap_width * col
        width_end = heatmap_width * (col+1)

        grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image

    return grid_image

# --------------------------------------------------------------------------------
# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
