#Copyright 2021 Huawei Technologies Co., Ltd
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

import os
from utils.timer import Timer
from utils.logger import Logger
from utils import utils
import time
import torch
if torch.__version__ >= "1.8":
    import torch_npu
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.measure import compare_ssim

def rgb2y_matlab(x):
    K = np.array([65.481, 128.553, 24.966]) / 255.0
    Y = 16 + np.matmul(x, K)
    return Y.astype(np.uint8)


def PSNR(im1, im2, use_y_channel=True):
    """Calculate PSNR score between im1 and im2
    --------------
    # Args
        - im1, im2: input byte RGB image, value range [0, 255]
        - use_y_channel: if convert im1 and im2 to illumination channel first
    """
    if use_y_channel:
        im1 = rgb2y_matlab(im1)
        im2 = rgb2y_matlab(im2)
    im1 = im1.astype(np.float)
    im2 = im2.astype(np.float)
    mse = np.mean(np.square(im1 - im2))
    return 10 * np.log10(255 ** 2 / mse)


def SSIM(gt_img, noise_img):
    """Calculate SSIM score between im1 and im2 in Y space
    -------------
    # Args
        - gt_img: ground truth image, byte RGB image
        - noise_img: image with noise, byte RGB image
    """
    gt_img = rgb2y_matlab(gt_img)
    noise_img = rgb2y_matlab(noise_img)

    ssim_score = compare_ssim(gt_img, noise_img, gaussian_weights=True,
                              sigma=1.5, use_sample_covariance=False)
    return ssim_score


def psnr_ssim_dir(gt_dir, test_dir):
    gt_img_list = sorted([x for x in sorted(os.listdir(gt_dir))])
    test_img_list = sorted([x for x in sorted(os.listdir(test_dir))])

    psnr_score = 0
    ssim_score = 0
    for gt_name, test_name in zip(gt_img_list, test_img_list):
        gt_img = Image.open(os.path.join(gt_dir, gt_name))
        test_img = Image.open(os.path.join(test_dir, test_name))
        gt_img = np.array(gt_img)
        test_img = np.array(test_img)
        psnr_score += PSNR(gt_img, test_img)
        ssim_score += SSIM(gt_img, test_img)
    return psnr_score / len(gt_img_list), ssim_score / len(gt_img_list)

if __name__ == '__main__':
    opt = TrainOptions().parse()
    t = opt.data_url
    opt.data_url = opt.data_url + "/img_align_celeba"
    flag = False
    device_id=int(os.environ['ASCEND_DEVICE_ID'])
    CALCULATE_DEVICE = "npu:{}".format(device_id)
    print("use ", CALCULATE_DEVICE)
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    logger = Logger(opt)
    timer = Timer()

    single_epoch_iters = (dataset_size // opt.batch_size)
    total_iters = opt.total_epochs * single_epoch_iters 
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters
    start_iter = opt.resume_iter
    print('Start training from epoch: {:05d}; iter: {:07d}'.format(opt.resume_epoch, opt.resume_iter))
    for epoch in range(opt.resume_epoch, opt.total_epochs + 1):    
        for i, data in enumerate(dataset, start=start_iter):
            cur_iters += 1
            logger.set_current_iter(cur_iters)
            # =================== load data ===============
            model.set_input(data, cur_iters)
            timer.update_time('DataTime')
            # =================== model train ===============
            model.forward(), timer.update_time('Forward')
            model.optimize_parameters(), timer.update_time('Backward')
            loss = model.get_current_losses()
            loss.update(model.get_lr())
            logger.record_losses(loss)
            # =================== save model and visualize ===============
            if cur_iters % opt.print_freq == 0:
                print('Model log directory: {}'.format(opt.expr_dir))
                epoch_progress = '{:03d}|{:05d}/{:05d}'.format(epoch, i, single_epoch_iters)
                logger.printIterSummary(epoch_progress, cur_iters, total_iters, timer)
    
            if cur_iters % opt.visual_freq == 0:
                visual_imgs = model.get_current_visuals()
                logger.record_images(visual_imgs)

            info = {'resume_epoch': epoch, 'resume_iter': i+1}
            if cur_iters == 126600:
                save_suffix = 'iter_%d' % cur_iters
                print('saving current model (epoch %d, iters %d)' % (epoch, cur_iters))
                model.save_networks(save_suffix, info)
                flag = True
                break
        if flag:
            break
            if opt.debug: break
        if opt.debug and epoch > 5: exit()
    logger.close()

# test
    opt.dataset_name = "single"
    opt.data_url = t + "/test_dirs/Helen_test_DIC/LR"
    opt.pretrain_model_path = opt.train_url + 'sparnet-best_model.pth'
    opt.save_as_dir = opt.train_url + "Face-SPARNet/SPARNet_S16_V4_Attn2D/"
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    if len(opt.pretrain_model_path):
        model.load_pretrain_model()
    else:
        model.setup(opt)  # regular setup: load and print networks; create schedulers

    if len(opt.save_as_dir):
        save_dir = opt.save_as_dir
    else:
        save_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
        if opt.load_iter > 0:  # load_iter is 0 by default
            save_dir = '{:s}_iter{:d}'.format(save_dir, opt.load_iter)
    os.makedirs(save_dir, exist_ok=True)

    print('creating result directory', save_dir)

    network = model.netG.to(CALCULATE_DEVICE)
    network.eval()

    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        inp = data['LR'].to(CALCULATE_DEVICE)
        with torch.no_grad():
            output_SR = network(inp).to(CALCULATE_DEVICE)
        img_path = data['LR_paths']  # get image paths
        output_sr_img = utils.tensor_to_img(output_SR, normal=True)

        save_path = os.path.join(save_dir, img_path[0].split('/')[-1])
        save_img = Image.fromarray(output_sr_img)
        save_img.save(save_path)

    # python psnr_ssim.py
    gt_dir = t + '/test_dirs/Helen_test_DIC/HR'
    test_dirs = [
            opt.train_url + '/results_helen/SPARNet_S16_V4_Attn2D',
            ]
    for td in test_dirs:
        result = psnr_ssim_dir(td, gt_dir)
        print(td, result)