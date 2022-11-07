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

### Copyright (C) 2020 Roy Or-El. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import scipy # this is to prevent a potential error caused by importing torch before scipy (happens due to a bad combination of torch & scipy versions)
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
# from util.visualizer import Visualizer
import os
import numpy as np
import torch
from pdb import set_trace as st
from tqdm import tqdm
# from torchvision import models
# from torchsummary import summary


def print_current_errors(opt, epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

def train(opt):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    if opt.continue_train:
        if opt.which_epoch == 'latest':
            try:
                start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
            except:
                start_epoch, epoch_iter = 1, 0
        else:
            start_epoch, epoch_iter = int(opt.which_epoch), 0

        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
        for update_point in opt.decay_epochs:
            if start_epoch < update_point:
                break

            opt.lr *= opt.decay_gamma
    else:
        start_epoch, epoch_iter = 0, 0

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    # print(model)
    # summary(model, (6, 3, 256, 256))
    # visualizer = Visualizer(opt)

    total_steps = (start_epoch) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    bSize = opt.batchSize

    #in case there's no display sample one image from each class to test after every epoch
    if opt.display_id == 0:
        dataset.dataset.set_sample_mode(True)
        dataset.num_workers = 1
        for i, data in enumerate(dataset):
            if i*opt.batchSize >= opt.numClasses:
                break
            if i == 0:
                sample_data = data
            else:
                for key, value in data.items():
                    if torch.is_tensor(data[key]):
                        sample_data[key] = torch.cat((sample_data[key], data[key]), 0)
                    else:
                        sample_data[key] = sample_data[key] + data[key]
        dataset.num_workers = opt.nThreads
        dataset.dataset.set_sample_mode(False)

    for epoch in tqdm(range(start_epoch, opt.epochs)):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = 0
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = (total_steps % opt.display_freq == display_delta) and (opt.display_id > 0)

            ############## Network Pass ########################
            model.set_inputs(data)
            disc_losses = model.update_D()
            gen_losses, gen_in, gen_out, rec_out, cyc_out = model.update_G(infer=save_fake)
            loss_dict = dict(gen_losses, **disc_losses)
            ##################################################

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.item() if not (isinstance(v, float) or isinstance(v, int)) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                # visualizer.print_current_errors(epoch+1, epoch_iter, errors, t)
                print_current_errors(opt,epoch+1, epoch_iter, errors, t)
                if opt.display_id > 0:
                    pass
                    # visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            # ### display output images
            # if save_fake and opt.display_id > 0:
            #     class_a_suffix = ' class {}'.format(data['A_class'][0])
            #     class_b_suffix = ' class {}'.format(data['B_class'][0])
            #     classes = None
            #
            #     visuals = OrderedDict()
            #     visuals_A = OrderedDict([('real image' + class_a_suffix, util.tensor2im(gen_in.data[0]))])
            #     visuals_B = OrderedDict([('real image' + class_b_suffix, util.tensor2im(gen_in.data[bSize]))])
            #
            #     A_out_vis = OrderedDict([('synthesized image' + class_b_suffix, util.tensor2im(gen_out.data[0]))])
            #     B_out_vis = OrderedDict([('synthesized image' + class_a_suffix, util.tensor2im(gen_out.data[bSize]))])
            #     if opt.lambda_rec > 0:
            #         A_out_vis.update([('reconstructed image' + class_a_suffix, util.tensor2im(rec_out.data[0]))])
            #         B_out_vis.update([('reconstructed image' + class_b_suffix, util.tensor2im(rec_out.data[bSize]))])
            #     if opt.lambda_cyc > 0:
            #         A_out_vis.update([('cycled image' + class_a_suffix, util.tensor2im(cyc_out.data[0]))])
            #         B_out_vis.update([('cycled image' + class_b_suffix, util.tensor2im(cyc_out.data[bSize]))])
            #
            #     visuals_A.update(A_out_vis)
            #     visuals_B.update(B_out_vis)
            #     visuals.update(visuals_A)
            #     visuals.update(visuals_B)
            #
            #     # ncols = len(visuals_A)
            #     # visualizer.display_current_results(visuals, epoch, classes, ncols)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch+1, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
                if opt.display_id == 0:
                    model.eval()
                    # visuals = model.inference(sample_data)
                    # visualizer.save_matrix_image(visuals, 'latest')
                    model.train()

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch+1, opt.epochs, time.time() - epoch_start_time))

        ### save model for this epoch
        if (epoch+1) % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch+1, total_steps))
            model.save('latest')
            model.save(epoch+1)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
            if opt.display_id == 0:
                model.eval()
                # visuals = model.inference(sample_data)
                # visualizer.save_matrix_image(visuals, epoch+1)
                model.train()

        ### multiply learning rate by opt.decay_gamma after certain iterations
        if (epoch+1) in opt.decay_epochs:
            model.update_learning_rate()

if __name__ == "__main__":
    opt = TrainOptions().parse()
    train(opt)
