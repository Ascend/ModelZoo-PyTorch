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

#from torch.utils.tensorboard import SummaryWriter

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.utils as utils
from lib.train.train_covid import train, validation


def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)
    name_model = args.model + "_" + args.dataset_name + "_" + utils.datestr()

    # TODO visual3D_temp.Basewriter package
    #writer = SummaryWriter(log_dir='../runs/' + name_model, comment=name_model)

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='.././datasets')
    model, optimizer = medzoo.create_model(args)

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    print("START TRAINING...")
    for epoch in range(1, args.nEpochs + 1):
        train(args, model, training_generator, optimizer, epoch, writer)
        val_metrics, confusion_matrix = validation(args, model, val_generator, epoch, writer)

        # utils.write_train_val_score(writer, epoch, train_stats, val_stats)
        #
        # model.save_checkpoint(args.save, epoch, val_stats[0], optimizer=optimizer)
        #


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="COVIDx")
    parser.add_argument('--dim', nargs="+", type=int, default=(32, 32, 32))
    parser.add_argument('--nEpochs', type=int, default=250)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--samples_train', type=int, default=10)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--inChannels', type=int, default=2)
    parser.add_argument('--inModalities', type=int, default=2)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='CNN',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
