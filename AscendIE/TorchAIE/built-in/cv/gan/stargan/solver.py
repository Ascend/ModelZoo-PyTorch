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

from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

import torch_aie
from torch_aie import _enums


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cpu') # revise
        # self.device = "npu:0"

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])            
        self.G.to(self.device)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.pth'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to("npu:0"))
        return c_trg_list


    def test(self, ts_model_path):
        """Translate images using StarGAN trained on a single dataset."""
        if not os.path.exists('./bin/attr'):
            os.makedirs('./bin/attr')
        if not os.path.exists('./bin/img'):
            os.makedirs('./bin/img')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        ts_model = torch.jit.load(ts_model_path)
        input_info = [torch_aie.Input((16, 3, 128, 128)), torch_aie.Input((16, 5))]
        torch_aie.set_device(0)
        print("start_compile")
        torchaie_model = torch_aie.compile(
            ts_model,
            inputs=input_info,
            precision_policy=torch_aie.PrecisionPolicy.PREF_FP32, # _enums.PrecisionPolicy.FP32
            soc_version='Ascend310P3'
        )
        print("end_compile")
        torchaie_model.eval()

        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.celeba_loader
        
        with torch.no_grad():
            cnt = 0
            print("data_loader has len:", len(data_loader))
            for i, (x_real, c_org) in enumerate(data_loader):
                
                print("current i is:", i)

                # Prepare input images and target domain labels.
                x_real_cpu = x_real.to("cpu")
                x_real_npu = x_real.to("npu:0")
                print("x_real_cpu's dtype is:", x_real_cpu.dtype)
                print("x_real_npu's dtype is:", x_real_npu.dtype)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                # x_fake_list = [x_real]
                x_fake_list = []
                for c_trg in c_trg_list:
                    c_trg_cpu = c_trg.to("cpu")
                    print("c_trg_cpu's dtype is:", c_trg_cpu.dtype)
                    print("c_trg_list has length", len(c_trg_list))
                    x_new = ts_model(x_real_cpu, c_trg_cpu)
                    # x_new = self.G(x_real, c_trg)
                    # x_new = x_new.to("cpu")
                    
                    c_trg_npu = c_trg.to("npu:0")
                    x_new_aie = torchaie_model(x_real_npu, c_trg_npu)
                    x_new_aie = x_new_aie.to("cpu")
                    
                    print("11111111111111 Cosine Similarity between torch and torch_aie is: ",torch.cosine_similarity(x_new.reshape(1,-1), x_new_aie.reshape(1,-1)))
                    
                    x_fake_list.append(x_new_aie)
                    
                    # print("111111111", res.shape)
                    # x_fake_list.append(res)
                    
                    # print("x_real.shape", x_real.numpy().shape)
                    # print("x_real.dtype", x_real.dtype)
                    # print("c_trg.shape", c_trg.numpy())
                    # print("c_trg.dtype", c_trg.dtype)
                    # x_real.numpy().tofile("./bin/img" + "/%d.bin" % cnt)
                    # attr_bin_filepath = "./bin/attr" + "/%d.bin" % cnt
                    # c_trg.numpy().tofile(attr_bin_filepath)
                    # # print('Saved bin into ./bin/img/%d.bin...' % cnt)
                    # # print('Saved bin into ./bin/attr/%d.bin...' % cnt)
                    
                    # attr_np_arr = np.fromfile(attr_bin_filepath)
                    # print("1111111111111",attr_np_arr)
                    cnt = cnt + 1

                # Save the translated images.
                # x_concat = torch.cat(x_fake_list, dim=3)
                # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                
                x_concat = torch.cat(x_fake_list, dim=3)
                print("22222222222222", x_concat.shape)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
                if cnt >= 319: break