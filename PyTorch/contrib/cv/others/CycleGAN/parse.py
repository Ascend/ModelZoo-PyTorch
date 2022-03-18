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
import argparse
import torch


class parse_args():
    def __init__(self, isTrain=True, isTest=False):
        self.isTrain = isTrain
        self.isTest = isTest
        self.parser = argparse.ArgumentParser(description='Pytorch CycleGAN training')

    def initialize(self):
        parser = self.parser
        parser.add_argument('--model_ga_path', type=str,
                            default='./checkpoints_8pbs1/maps_cycle_gan/175_pu0_net_G_A.pth',
                            help='path for modelga')
        parser.add_argument('--model_gb_path', type=str,
                            default='./checkpoints_8pbs1/maps_cycle_gan/175_pu0_net_G_B.pth',
                            help='path for modelga')
        parser.add_argument('--prof', type=int, default=1, help='whether to get prof file')
        parser.add_argument('--num_epoch', type=int, default=240, help='whether to get prof file1')
        parser.add_argument('--line_scale', type=float, default=2, help='whether to get prof file1')
        parser.add_argument('--num_epoch_start', type=int, default=0, help='whether to get prof file1')
        parser.add_argument('--loadweight', default='latest', help='whether to get prof file1')
        parser.add_argument('--prof_file', type=str, default='./output.prof', help='whether to get prof file')
        parser.add_argument('--log_path', type=str, default='gpu1p.txt', help='whether to get prof file')
        parser.add_argument('--multiprocessing_distributed', type=int, default=1,
                            help='Use multi-processing distributed training to launch,if it is eaqul to 1 or  more than ,start to  npu/gpu Multi-card training ')
        parser.add_argument('--world_size', type=int, default=1, help='word__size')
        parser.add_argument('--distributed', type=int, default=1,
                            help='whether to use distributed to fastern training,if it is eaqul to 1 or  more than ,start to  npu/gpu Multi-card training')
        parser.add_argument('--rank', default=0, type=int, help='rank')
        parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
        parser.add_argument('--npu', type=int, default=0, help='whether to use npu to fastern training')
        parser.add_argument('--pu_ids', type=str, default='0,1',
                            help='gpu ids(npu ids): e.g. 0  0,1,2, 0,2. use -1 for CPU')

        parser.add_argument('--isapex', default=True, help='whether to use apex to fastern training')
        parser.add_argument('--apex_type', type=str, default="O1", help='O0,O1,O2,O3')
        parser.add_argument('--loss_scale', default=None, help='loss_scale:1,128,dynamic')
        parser.add_argument('--dataroot', type=str, default='./datasets/maps',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='maps_cycle_gan',
                            help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--checkpoints_dir', type=str, default='./re_checkpoints2p_bs1',
                            help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan',
                            help='chooses which model to use. [cycle_gan| pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel].'
                                 ' The basic model is a 70x70 PatchGAN. n_layers allows you to '
                                 'specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks',
                            help='specify generator architecture [resnet_9blocks | resnet_6blocks | '
                                 'unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned',
                            help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory '
                                 'contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | '
                                 'scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_'
                                 '[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument(
            "--cache-dataset",
            dest="cache_dataset",
            help="Cache the datasets for quicker initialization. It also serializes the transforms",
            action="store_true",
        )
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if (self.isTrain):
            # network saving and loading parameters
            parser.add_argument('--display_freq', type=int, default=400,
                                help='frequency of showing training results on screen')
            parser.add_argument('--display_ncols', type=int, default=4,
                                help='if positive, display all images in a single visdom web panel with '
                                     'certain number of images per row.')
            parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
            parser.add_argument('--display_server', type=str, default="http://localhost",
                                help='visdom server of the web display')
            parser.add_argument('--display_env', type=str, default='main',
                                help='visdom display environment name (default is "main")')
            parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
            parser.add_argument('--update_html_freq', type=int, default=1000,
                                help='frequency of saving training results to html')
            parser.add_argument('--print_freq', type=int, default=100,
                                help='frequency of showing training results on console')
            parser.add_argument('--no_html', action='store_true',
                                help='do not save intermediate training results to ['
                                     'opt.checkpoints_dir]/[opt.name]/web/')
            # network saving and loading parameters
            parser.add_argument('--save_latest_freq', type=int, default=5000,
                                help='frequency of saving the latest results')
            parser.add_argument('--save_epoch_freq', type=int, default=5,
                                help='frequency of saving checkpoints at the end of epochs')
            parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
            parser.add_argument('--continue_train', action='store_true',
                                help='continue training: load the latest model')
            parser.add_argument('--epoch_count', type=int, default=1,
                                help='the starting epoch count, we save the model '
                                     'by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
            parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
            # training parameters
            parser.add_argument('--n_epochs', type=int, default=100,
                                help='number of epochs with the initial learning rate')
            parser.add_argument('--n_epochs_decay', type=int, default=100,
                                help='number of epochs to linearly decay learning rate to zero')
            parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
            parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
            parser.add_argument('--gan_mode', type=str, default='lsgan',
                                help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is'
                                     ' the cross-entropy objective used in the original GAN paper.')
            parser.add_argument('--pool_size', type=int, default=50,
                                help='the size of image buffer that stores previously generated images')
            parser.add_argument('--lr_policy', type=str, default='linear',
                                help='learning rate policy. [linear | step | plateau | cosine]')
            parser.add_argument('--lr_decay_iters', type=int, default=50,
                                help='multiply by a gamma every lr_decay_iters iterations')
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of'
                                     ' scaling the weight of the identity mapping loss. For example, if the weight of'
                                     ' the identity loss should be 10 times smaller than the weight of the '
                                     'reconstruction loss, please set lambda_identity = 0.1')
        parser = parser.parse_args()
        parser.process_device_map = self.device_id_to_process_device_map(parser.pu_ids)
        return parser

    def device_id_to_process_device_map(self, device_list):
        devices = device_list.split(",")
        devices = [int(x) for x in devices]
        devices.sort()

        process_device_map = dict()
        for process_id, device_id in enumerate(devices):
            process_device_map[process_id] = device_id
        return process_device_map

    def change_parser(self, isTrain=True, isTest=False):
        self.isTest = isTest
        self.isTrain = isTrain
        self.parser = None
        return self.initialize()

    def printParser(self):
        pasers = self.parser.parse_args()
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(pasers).items()):
            comment = ''
            default = self.parser.get_default(k)
            # if v != default:
            #     comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def init_distributed_mode(self, ngpus_per_node, gpu):
        opt = self.parser.parse_args()
        if opt.multiprocessing_distributed >= 1:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        if (opt.npu < 1):
            torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=opt.world_size,
                                                 rank=opt.rank)
        elif (opt.npu >= 1):
            torch.distributed.init_process_group(backend='hccl', world_size=opt.world_size, rank=opt.rank)
