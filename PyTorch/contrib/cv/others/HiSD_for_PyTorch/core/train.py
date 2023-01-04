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
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
from utils import get_data_iters, prepare_sub_folder, get_config,write_2images,write_loss
import argparse
from trainer import HiSD_Trainer
import torch
if torch.__version__ >= '1.8.1':		
    import torch_npu
import os
import sys
import shutil
import random
from utils import AverageMeter
from torch.backends import cudnn
import matplotlib.pyplot as plt
import time



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/celeba-hq.yaml", help='Path to the config file.')
parser.add_argument('--batchsize', type=int, default=8, help="outputs path")
parser.add_argument('--total_iterations', type=int, default=40, help="outputs path")
#parser.add_argument('--data_path', type=str, default='', help="outputs path")
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--gpus', type=int, default=2)
opts = parser.parse_args()

# For fast training
cudnn.benchmark = True
CALCULATE_DEVICE = 'npu:{}'.format(opts.gpus)
torch.npu.set_device(CALCULATE_DEVICE)
# Load experiment setting
config = get_config(opts.config)
total_iterations = opts.total_iterations
config['batch_size'] = opts.batchsize


# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

# Setup model
trainer = HiSD_Trainer(config)
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0


trainer.npu(opts.gpus)

# Setup data loader
train_iters = get_data_iters(config, opts.gpus)
tags = list(range(len(train_iters)))

#start = time.time()
tot_time = AverageMeter('Time', ':6.3f')
G_LOSS = []
D_LOSS = []
iteration_ = []
f1 = open("G_loss.txt", 'w')
f2 = open("D_loss.txt", 'w')

while True:
    """
    i: tag
    j: source attribute, j_trg: target attribute
    x: image, y: tag-irrelevant conditions
    """
    torch.npu.synchronize()
    end = time.time()
    
    i = random.sample(tags, 1)[0]
    j, j_trg = random.sample(list(range(len(train_iters[i]))), 2) 
    x, y = train_iters[i][j].next()
    train_iters[i][j].preload()

    # Train processing
    G_adv, G_sty, G_rec, D_adv = trainer.update(x, y, i, j, j_trg)
    torch.npu.synchronize()
    current_batch_time = time.time() - end
    #print("sec/step : {}".format(current_batch_time))
    
    if iterations > 20:
        tot_time.update(current_batch_time)    
        print("sec/step : {}".format(current_batch_time))
        #print(tot_time.avg) 

    # if (iterations + 1) % config['image_save_iter'] == 0:
    #     for i in range(len(train_iters)):
    #         j, j_trg = random.sample(list(range(len(train_iters[i]))), 2) 

    #         x, _ = train_iters[i][j].next()
    #         x_trg, _ = train_iters[i][j_trg].next()
    #         train_iters[i][j].preload()
    #         train_iters[i][j_trg].preload()

    #         test_image_outputs = trainer.sample(x, x_trg, j, j_trg, i)
    #         write_2images(test_image_outputs,
    #                       config['batch_size'], 
    #                       image_directory, 'sample_%08d_%s_%s_to_%s' % (iterations + 1, config['tags'][i]['name'], config['tags'][i]['attributes'][j]['name'], config['tags'][i]['attributes'][j_trg]['name']))

    
    if (iterations + 1) % config['log_iter'] == 0:
        loss_t  = G_adv+G_sty+G_rec
        print("Gen Loss:{:.3f}, Dis Loss:{:.3f}".format(loss_t,D_adv))
        if iterations > 20:
            print('Total FPS = {:.2f}\t'.format(config['batch_size'] / tot_time.avg))
        # iteration_.append(iterations+1)
        # G_LOSS.append(int(loss_t))
        # D_LOSS.append(int(D_adv))
        # y1=G_LOSS
        # y2=D_LOSS
        # plt.plot(iteration_,y1,'b',label='loss')
        # plt.title('loss vs. iteration')
        # plt.xlabel('iterations')
        # plt.ylabel('g loss')
        # plt.savefig("accuracy_G_loss.jpg")
        # plt.close()
        # plt.plot(iteration_,y2,'b',label='111')
        # plt.title('loss vs. iteration')
        # plt.xlabel('iterations')
        # plt.ylabel('d loss')
        # plt.savefig("accuracy_D_loss.jpg")
        # plt.close()
        
        f1.write('iterations '+str(iterations)+": "+str(loss_t)+"\r\n")
        f2.write('iterations '+str(iterations)+": "+str(D_adv)+"\r\n")


    if (iterations + 1) == total_iterations:
        print('Finish training!')
        print('Total FPS = {:.2f}\t'.format(config['batch_size'] / tot_time.avg))
        trainer.save(checkpoint_directory, iterations)
        # y1=G_LOSS
        # y2=D_LOSS
        # plt.plot(iteration_,y1,'b',label='loss')
        # plt.title('loss vs. iteration')
        # plt.xlabel('iterations')
        # plt.ylabel('g loss')
        # plt.savefig("accuracy_G_loss.jpg")
        # plt.close()
        # plt.plot(iteration_,y2,'b',label='111')
        # plt.title('loss vs. iteration')
        # plt.xlabel('iterations')
        # plt.ylabel('d loss')
        # plt.savefig("accuracy_D_loss.jpg")
        # plt.close()
        f1.close()
        f2.close()
        exit()
    iterations += 1



   

