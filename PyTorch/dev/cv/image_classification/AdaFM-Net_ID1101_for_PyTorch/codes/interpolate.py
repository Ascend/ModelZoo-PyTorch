#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import os
import logging
import argparse
import numpy as np
from collections import OrderedDict
import torch

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)
stride = opt['interpolate_stride'] if opt['interpolate_stride'] is not None else 0.1
model_dict = torch.load(opt['path']['pretrain_model_G'])


for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nModulating [{:s}]...'.format(test_set_name))
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)
    need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

    for coef in np.arange(0.0, 1.01, stride):
        print('setting coef to {:.2f}'.format(coef))

        interp_dict = model_dict.copy()
        for k, v in model_dict.items():
            if k.find('transformer') >= 0:
                interp_dict[k] = v * coef
        model.update(interp_dict)

        # load the test data
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        for data in test_loader:
            model.feed_data(data, need_HR=need_HR)
            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_dir = os.path.join(dataset_dir, img_name)
            util.mkdir(img_dir)

            model.test()

            visuals = model.get_current_visuals(need_HR=need_HR)

            sr_img = util.tensor2img(visuals['SR'])  # uint8

            # save images
            img_part_name = '_coef_{:.2f}.png'.format(coef)
            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(img_dir, img_name + suffix + img_part_name)
            else:
                save_img_path = os.path.join(img_dir, img_name + img_part_name)
            util.save_img(sr_img, save_img_path)

            # calculate PSNR and SSIM
            if need_HR:
                gt_img = util.tensor2img(visuals['HR'])
                gt_img = gt_img / 255.
                sr_img = sr_img / 255.

                crop_size = opt['crop_size']
                if crop_size > 0:
                    sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

                psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)
                ssim = util.calculate_ssim(sr_img * 255, gt_img * 255)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_size > 0:
                        sr_img_y = sr_img_y[crop_size:-crop_size, crop_size:-crop_size]
                        gt_img_y = gt_img_y[crop_size:-crop_size, crop_size:-crop_size]
                    psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                    ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'\
                        .format(img_name, psnr, ssim, psnr_y, ssim_y))
                else:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            else:
                logger.info(img_name)

        if need_HR:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'\
                    .format(test_set_name, ave_psnr, ave_ssim))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'\
                    .format(ave_psnr_y, ave_ssim_y))
