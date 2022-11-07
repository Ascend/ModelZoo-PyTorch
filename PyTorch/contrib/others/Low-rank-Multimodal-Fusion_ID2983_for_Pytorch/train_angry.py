# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
import os

from model import LMF
from utils import total, load_iemocap
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import pandas as pd
import os
import random
import apex
from apex import amp
import argparse
import torch
if torch.__version__ >= "1.8":
     import torch_npu
print(torch.__version__)
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import time

DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != DEVICE:
    DEVICE = torch.npu.set_device(f'npu:{DEVICE}')

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 预处理数据以及训练模型
def display(F1_score, accuracy_score):
    print("F1_score on test set is {}".format(F1_score))
    print("Accuracy score on test set is {}".format(accuracy_score))

def main(options):
    emotion = options['emotion']
    if emotion == b'angry':
        index=0
        num = 223
    elif emotion == b'sad':
        index=1
        num = 650
    elif emotion == b'happy':
        index=2
        num = 432
    else:
        index=3
        num = 504
    DTYPE = torch.FloatTensor
    LONG = torch.LongTensor
    setup_seed(num)
    # parse the input args
    epochs = options['train_epochs']
    times=options['train_time']
    # data_path = options['data_path']
    data_path = options['data_url']
    model_path = options['model_path']
    output_path = options['output_path']
    signiture = options['signiture']
    patience = options['patience']
    # emotion = options['emotion']
    output_dim = options['output_dim']
    para_path=options['para_path']
    # index=options['index']


    # prepare the paths for storing models and outputs
    best_model_path = os.path.join(
        model_path, "best_model_{}_{}.pt".format(signiture, emotion))
    model_path = os.path.join(
        model_path, "model_{}_{}.pt".format(signiture, emotion))
    # prepare the paths for storing models and outputs
    output_path = os.path.join(
        output_path, "results_{}_{}.csv".format(signiture, emotion))
    print("emotion {}".format(emotion))
    print("Temp location for models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_set, valid_set, test_set, input_dims = load_iemocap(data_path, emotion)

    params = dict()
    # params_file=open("param.csv","r")
    rows=np.loadtxt(open(para_path,"rb"),delimiter=",",skiprows=0)
    


    params['audio_hidden'] =int(rows[index][0])
    params['video_hidden'] =int(rows[index][1])
    params['text_hidden'] =int(rows[index][2])
    params['audio_dropout'] =rows[index][3]
    params['video_dropout'] =rows[index][4]
    params['text_dropout'] =rows[index][5]
    params['factor_learning_rate'] =rows[index][6]
    params['learning_rate'] =rows[index][7]
    params['rank'] =int(rows[index][8])
    params['batch_size'] =int(rows[index][9])
    params['weight_decay'] =rows[index][10]


    seen_settings = set()
    best_now=0
    best_acc=0
    batch_sz = params['batch_size']
    print("batch_size is: {}".format(batch_sz))
    for i in range(times):
        if not os.path.isfile(output_path):
            with open(output_path, 'w+') as out:
                writer = csv.writer(out)
                writer.writerow(["audio_hidden", "video_hidden", 'text_hidden', 'audio_dropout', 'video_dropout', 'text_dropout',
                                'factor_learning_rate', 'learning_rate', 'rank', 'batch_size', 'weight_decay',
                                'Best Validation CrossEntropyLoss', 'Test CrossEntropyLoss', 'Test F1-score', 'Test Accuracy Score', 'num'])

        ahid =params['audio_hidden']
        vhid = params['video_hidden']
        thid = params['text_hidden']
        thid_2 = thid // 2
        adr = params['audio_dropout']
        vdr = params['video_dropout']
        tdr = params['text_dropout']
        factor_lr = params['factor_learning_rate']
        lr = params['learning_rate']
        r = params['rank']
        

        decay = params['weight_decay']
        model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
        model = model.to(DEVICE)
        DTYPE = torch.npu.FloatTensor
        LONG = torch.npu.LongTensor
        print("Model initialized")
        print(torch.npu.is_available())
        print(torch.npu.current_device())
        criterion = nn.CrossEntropyLoss(size_average=False)
        for p in model.parameters():
            p.data = p.data.npu()
        factors = list(model.parameters())[:3]
        other = list(model.parameters())[3:]
        #optimizer = optim.Adam([{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}], weight_decay=decay)
        optimizer = apex.optimizers.NpuFusedAdam([{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}], weight_decay=decay)
        model, optimizer = amp.initialize(model.to(DEVICE), optimizer, opt_level="O2", loss_scale=128)
        # model, optimizer = amp.initialize(model.to(DEVICE), optimizer, opt_level = "O2", keep_batchnorm_fp32 = True,loss_scale = "dynamic")
        complete = True
        min_valid_loss = float('Inf')
        train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True)
        valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
        test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
        curr_patience = patience
        TrainLossList=[]
        ValLossList=[]
        for e in range(epochs):

            model.train()
            model.zero_grad()
            avg_train_loss = 0.0
            for batch in train_iterator:
                model.zero_grad()
                startime = time.time()
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                try:
                    output = model(x_a, x_v, x_t)
                except ValueError as e:
                    print(x_a.data.shape)
                    print(x_v.data.shape)
                    print(x_t.data.shape)
                    raise e
            
                output=output.type(torch.float16)
                y=y.type(torch.float16)
                a=torch.max(y, 1)[1]
                loss = criterion(output,a)
                #loss.backward()
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
                avg_loss = loss.item()
                avg_train_loss += avg_loss / len(train_set)
                loss=avg_train_loss
                endTime = time.time()
                steptime = endTime - startime
                print("The steptime is: {}".format(steptime))
                optimizer.step()
            TrainLossList.append(avg_train_loss)

            print("avg_train_loss is: {}".format(avg_train_loss))

            # Terminate the training process if run into NaN
            if np.isnan(avg_train_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            model.eval()
            for batch in valid_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                output = model(x_a, x_v, x_t)
                output=output.type(torch.float16)
                y=y.type(torch.float16)
                a=torch.max(y, 1)[1]
                valid_loss = criterion(output,a)
                avg_valid_loss = valid_loss.item()

            y = y.cpu().data.numpy().reshape(-1, output_dim)

            if np.isnan(avg_valid_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            avg_valid_loss = avg_valid_loss / len(valid_set)

            ValLossList.append(avg_valid_loss)
            print("Validation loss is: {}".format(avg_valid_loss))
            if (avg_valid_loss < min_valid_loss):
                curr_patience = patience
                min_valid_loss = avg_valid_loss
                torch.save(model.state_dict(), model_path)
                print("Found new best model, saving to disk...")
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break
            print("\n\n")
        plt.figure(1)
        plt.title(str(emotion))
        plt.plot(list(range(len(TrainLossList))), TrainLossList, label="TrainLoss")
        plt.plot(list(range(len(ValLossList))), ValLossList, label="ValLoss")
        plt.savefig(str(emotion) + ".png")
        plt.close(1)


        if complete:

            best_model = torch.load(model_path)
            # best_model.eval()
            model.load_state_dict(best_model)
            model.eval()
            for batch in test_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                output_test = model(x_a, x_v, x_t)
                output_test=output_test.type(torch.float16)
                y=y.type(torch.float16)
                a=torch.max(y, 1)[1]
                loss_test = criterion(output_test,a)
                test_loss = loss_test.item()
            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)
            test_loss = test_loss / len(test_set)

            # these are the needed metrics
            all_true_label = np.argmax(y,axis=1)
            all_predicted_label = np.argmax(output_test,axis=1)

            F1_score = f1_score(all_true_label, all_predicted_label, average='weighted')
            acc_score = accuracy_score(all_true_label, all_predicted_label)
            if F1_score>best_now:
                best_now= F1_score
                best_acc=acc_score
                torch.save(model.state_dict(), best_model_path)



    with open(output_path, 'a+') as out:
        writer = csv.writer(out)
        min_valid_loss = np.array(min_valid_loss, dtype=np.float16)
        min_valid_loss = np.array(min_valid_loss, dtype=np.float16)
        writer.writerow([ahid, vhid, thid, adr, vdr, tdr, factor_lr, lr, r, batch_sz, decay,
                        min_valid_loss, test_loss, best_now,best_acc, num])
    display(best_now,best_acc)


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--train_epochs', dest='train_epochs', type=int, default=500)
    OPTIONS.add_argument('--train_time', dest='train_time', type=int, default=100)
    OPTIONS.add_argument('--output_dim', dest='output_dim', type=int, default=2)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='')
    OPTIONS.add_argument('--data_path', dest='data_path',
                         type=str, default='data/')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results')
    OPTIONS.add_argument("--para_path", type=str, default="1")
    OPTIONS.add_argument("--data_url", type=str, default="./data/")
    OPTIONS.add_argument("--train_url", type=str,default="./outputs/")
    OPTIONS.add_argument("--emotion", type=str, default=b"neutral")
    PARAMS = vars(OPTIONS.parse_args())

    main(PARAMS)