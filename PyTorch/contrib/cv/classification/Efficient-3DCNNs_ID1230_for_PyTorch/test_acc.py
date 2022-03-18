# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
from run.spatial_transforms import *
from run.dataset import get_train_loader, get_val_loader, get_test_loader
from run.utils import adjust_learning_rate, save_checkpoint, Logger, AverageMeter, opt_preprocess
from run.test import test
from run.getmodel import generate_model, model_load_pretrained, replacemudule

def final_test(opt):
    # get model
    if torch.cuda.is_available():
        opt.gpu_or_npu = 'gpu'
        opt.device = torch.device("cuda:0")
        torch.cuda.set_device(opt.device)
        print('gpu...')
    else:
        opt.gpu_or_npu = 'npu'
        opt.device = torch.device("npu:0")
        torch.npu.set_device(opt.device)
        print('npu...')

    # load best weight
    if not opt.no_train or not opt.no_val: # train valid test
        bestweight_path = os.path.join(opt.root_path, opt.result_path+'/ucf101_mobilenetv2_1.0x_RGB_16_best.pth')
    else:
        bestweight_path = opt.resume_path # only test set

    model = generate_model(opt)
    model, parameters = model_load_pretrained(opt, model)

    if not opt.no_drive:
        model = model.to(opt.device)

    checkpoint = torch.load(bestweight_path)

    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(replacemudule(checkpoint['state_dict']))

    test_logger = Logger(os.path.join(opt.result_path, 'test.log'), ['top1 acc'])
    test_data, test_loader = get_test_loader(opt)
    test(test_loader, model, opt, test_data.class_names, test_logger, device_ids=0)

