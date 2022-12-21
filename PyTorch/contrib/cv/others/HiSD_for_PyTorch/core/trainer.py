# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
"""
from networks import Gen, Dis
from utils import weights_init, get_model_list
import torch
import torch_npu
import torch.nn as nn
import os
import torch.nn.functional as F
import copy
from apex import amp

def update_average(model_tgt, model_src, beta=0.99):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

class HiSD(nn.Module):
    def __init__(self, hyperparameters):
        super(HiSD, self).__init__()
        self.gen = Gen(hyperparameters)
        self.dis = Dis(hyperparameters)

        self.noise_dim = hyperparameters['noise_dim']
        self.hyperparameters = hyperparameters

    def forward(self, args, mode):
        if mode == 'gen':
            return self.gen_losses(*args)
        elif mode == 'dis':
            return self.dis_losses(*args)
        else:
            pass

    def gen_losses(self, x, y, i, j, j_trg):
        batch = x.size(0)

        # non-translation path
        e = self.gen.encode(x)
        x_rec = self.gen.decode(e)

        # self-translation path
        s = self.gen.extract(x, i)
        e_slf = self.gen.translate(e, s, i)
        x_slf = self.gen.decode(e_slf)

        # cycle-translation path
        ## translate
        s_trg = self.gen.map(torch.randn(batch, self.noise_dim).npu(), i, j_trg)
        e_trg = self.gen.translate(e, s_trg, i)
        x_trg = self.gen.decode(e_trg)
        ## cycle-back
        e_trg_rec = self.gen.encode(x_trg)
        s_trg_rec = self.gen.extract(x_trg, i) 
        e_cyc = self.gen.translate(e_trg_rec, s, i)
        x_cyc = self.gen.decode(e_cyc)

        # Added style in discriminator (ALI, Adversarially Learned Inference)
        # has not been added into the submission now, 
        # which helps disentangle the extracted style.
        # I will add this part in the next version.
        # To stable the training and avoid training crash, detaching is necessary.
        # Adding ALI will possibly make the metrics different from the paper,
        # but I do think this version would be better.
        loss_gen_adv = self.dis.calc_gen_loss_real(x, s, y, i, j) + \
                       self.dis.calc_gen_loss_fake_trg(x_trg, s_trg.detach(), y, i, j_trg) + \
                       self.dis.calc_gen_loss_fake_cyc(x_cyc, s.detach(), y, i, j) 

        loss_gen_sty = F.l1_loss(s_trg_rec, s_trg)

        loss_gen_rec = F.l1_loss(x_rec, x) + \
                       F.l1_loss(x_slf, x) + \
                       F.l1_loss(x_cyc, x)

        loss_gen_total = self.hyperparameters['adv_w'] * loss_gen_adv + \
                         self.hyperparameters['sty_w'] * loss_gen_sty + \
                         self.hyperparameters['rec_w'] * loss_gen_rec

        # loss_gen_total.backward()

        return loss_gen_total,loss_gen_adv, loss_gen_sty, loss_gen_rec, \
        x_trg.detach(), x_cyc.detach(), s.detach(), s_trg.detach()

    def dis_losses(self, x, x_trg, x_cyc, s, s_trg, y, i, j, j_trg):

        loss_dis_adv = self.dis.calc_dis_loss_real(x, s, y, i, j) + \
                       self.dis.calc_dis_loss_fake_trg(x_trg, s_trg, y, i, j_trg) + \
                       self.dis.calc_dis_loss_fake_cyc(x_cyc, s, y, i, j) 
        # loss_dis_adv.backward()

        return loss_dis_adv

class HiSD_Trainer(nn.Module):
    def __init__(self, hyperparameters, multi_gpus=False):
        super(HiSD_Trainer, self).__init__()
        # Initiate the networks
        self.multi_gpus = multi_gpus
        self.models = HiSD(hyperparameters)
        self.use_amp = True
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        self.dis_opt = torch.optim.Adam(self.models.dis.parameters(),
                                        lr=hyperparameters['lr_dis'], betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])

        self.gen_opt = torch.optim.Adam([{'params': self.models.gen.encoder.parameters()},
                                         {'params': self.models.gen.translators.parameters()},
                                         {'params': self.models.gen.extractors.parameters()},
                                         {'params': self.models.gen.decoder.parameters()},
                                         # Different LR for mappers.
                                         {'params': self.models.gen.mappers.parameters(), 'lr': hyperparameters['lr_gen_mappers']},
                                        ],
                                        lr=hyperparameters['lr_gen_others'], betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])

        self.models.dis = self.models.dis.npu()
        self.models.gen = self.models.gen.npu()
        if self.use_amp:
            self.models.dis, self.dis_opt = amp.initialize(self.models.dis, self.dis_opt, opt_level="O1", loss_scale=None )
            self.models.gen, self.gen_opt = amp.initialize(self.models.gen, self.gen_opt, opt_level="O1", loss_scale=None )
        self.apply(weights_init(hyperparameters['init']))

        # For historical average version of the generators
        self.models.gen_test = copy.deepcopy(self.models.gen)


    def update(self, x, y, i, j, j_trg):

        this_model = self.models.module if self.multi_gpus else self.models

        # gen 
        for p in this_model.dis.parameters():
            p.requires_grad = False
        for p in this_model.gen.parameters():
            p.requires_grad = True

        self.gen_opt.zero_grad()
        self.loss_gen_total,self.loss_gen_adv, self.loss_gen_sty, self.loss_gen_rec,x_trg, x_cyc, s, s_trg = self.models((x, y, i, j, j_trg), mode='gen')
        if self.use_amp:
            with amp.scale_loss(self.loss_gen_total, self.gen_opt) as gen_scaled_loss:
                gen_scaled_loss.backward()
        #self.gen_opt.step()
        #self.gen_opt.zero_grad()
        else:
            self.loss_gen_total.backward()
        # print("loss_gen_total: ",self.loss_gen_total.item())
            
        self.loss_gen_adv = self.loss_gen_adv.mean()
        self.loss_gen_sty = self.loss_gen_sty.mean()
        self.loss_gen_rec = self.loss_gen_rec.mean()
        
        nn.utils.clip_grad_norm_(this_model.gen.parameters(), 100)
        self.gen_opt.step()

        # dis
        for p in this_model.dis.parameters():
            p.requires_grad = True
        for p in this_model.gen.parameters():
            p.requires_grad = False

        self.dis_opt.zero_grad()

        self.loss_dis_adv = self.models((x, x_trg, x_cyc, s, s_trg, y, i, j, j_trg), mode='dis')
        if self.use_amp:
            with amp.scale_loss(self.loss_dis_adv, self.dis_opt) as dis_scaled_loss:
                dis_scaled_loss.backward()
        else:
            self.loss_dis_adv.backward()
        # print("loss_dis_adv: ",self.loss_dis_adv.item())
        # print('==============================')       
        self.loss_dis_adv = self.loss_dis_adv.mean()

        nn.utils.clip_grad_norm_(this_model.dis.parameters(), 100)
        self.dis_opt.step()

        update_average(this_model.gen_test, this_model.gen)

        return self.loss_gen_adv.item(), \
               self.loss_gen_sty.item(), \
               self.loss_gen_rec.item(), \
               self.loss_dis_adv.item()


    def sample(self, x, x_trg, j, j_trg, i):
        this_model = self.models.module if self.multi_gpus else self.models
        if True:
            gen = this_model.gen_test
        else:
            gen = this_model.gen

        out = [x]
        with torch.no_grad():

            e = gen.encode(x)

            # Latent-guided 1 
            z = torch.randn(1, gen.noise_dim).npu().repeat(x.size(0), 1)
            s_trg = gen.map(z, i, j_trg)
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg_]

            # Latent-guided 2
            z = torch.randn(1, gen.noise_dim).npu().repeat(x.size(0), 1)
            s_trg = gen.map(z, i, j_trg)
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg_]

            s_trg = gen.extract(x_trg, i)
            # Reference-guided 1: use x_trg[0, 1, ..., n] as reference
            x_trg_ = gen.decode(gen.translate(e, s_trg, i))
            out += [x_trg, x_trg_]

            # Reference-guided 2: use x_trg[n, n-1, ..., 0] as reference
            x_trg_ = gen.decode(gen.translate(e, s_trg.flip([0]), i))
            out += [x_trg.flip([0]), x_trg_]

        return out

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.models.gen.load_state_dict(state_dict['gen'])
        self.models.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.models.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.npu()
        
        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.npu()

        print('Resume from iteration %d' % iterations)
        return iterations
    

    def save(self, snapshot_dir, iterations):
        this_model = self.models.module if self.multi_gpus else self.models
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': this_model.gen.state_dict(), 'gen_test': this_model.gen_test.state_dict()}, gen_name)
        torch.save({'dis': this_model.dis.state_dict()}, dis_name)
       # torch.save({'dis': self.dis_opt.state_dict(), 
         #           'gen': self.gen_opt.state_dict()}, opt_name)
