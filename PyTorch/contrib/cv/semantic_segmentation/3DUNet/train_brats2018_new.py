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


# Python libraries
import argparse
import os
import torch
from apex import amp
#-------------------------
if torch.__version__ >= '1.8':
    import torch_npu
#-------------------------
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss
from lib.losses3D import VAEloss
from lib.utils.general import prepare_input

def main():
    args = get_arguments()
    print(args)
    if args.world_size > 1:
        torch.distributed.init_process_group(backend='hccl',  init_method="tcp://127.0.0.1:29999", world_size=args.world_size, rank=args.rank)
        print('torch.distributed.init_process_group done..', args.rank)
    torch.npu.set_device(args.rank)
    print('torch.npu.set_device(args.rank)..', args.rank)
    #torch.npu.set_device(CALCULATE_DEVICE)


    utils.reproducibility(args, args.rank)
    utils.make_dirs(args.save)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path=args.data_path)
    model, optimizer = medzoo.create_model(args)
    criterion = DiceLoss(classes=args.classes).npu()

    # if args.cuda:
    #     model = model.cuda()
    #---------------------------
    if args.device == 'npu':
        model = model.npu()
    #---------------------------
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=None, combine_grad=True)

    if args.world_size > 1:
       model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)

       
    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator)
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="brats2018")
    parser.add_argument('--data_path', type=str, default="./datasets")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=1024)
    parser.add_argument('--samples_val', type=int, default=128)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--inModalities', type=int, default=4)
    parser.add_argument('--threshold', default=0.00000000001, type=float)
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--cuda', action='store_true', default=False)
    
    parser.add_argument('--loadData', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='./runs/')
    parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')
    # dist
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    # amp
    parser.add_argument('--amp', action='store_true', default=False)

    parser.add_argument('--workers', type=int, default=8)

#-----------------------------------------------------------------------------------------
    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
#-----------------------------------------------------------------------------------------

    args = parser.parse_args()

    args.save = './saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)

    args.warm_up_epochs = int(args.nEpochs * 0.1)
    args.lr = args.lr * args.world_size


    return args


if __name__ == '__main__':
        
    main()
