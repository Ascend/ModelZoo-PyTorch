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
import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/root/ennengyang_space/Efficient-3DCNNs-master/', type=str, help='Root directory path of data')
    parser.add_argument('--video_path', default='annotation_UCF101/UCF-101-image', type=str, help='Directory path of Videos')
    parser.add_argument('--annotation_path', default='annotation_UCF101/ucf101_01.json', type=str, help='Annotation file path')
    # parser.add_argument('--resume_path', default='results/ucf101_mobilenetv2_1.0x_RGB_16_checkpoint.pth', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    # parser.add_argument('--pretrain_path', default='pretrain/kinetics_mobilenetv2_1.0x_RGB_16_best_dp.pth', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--ft_portion', default='complete', type=str, help='The portion of the model to apply fine tuning, either complete or last_layer')

    parser.add_argument('--no_drive', action='store_true', help='If true, cuda or npu is not used.')
    parser.set_defaults(no_drive=False)
    parser.add_argument('--gpu_or_npu',  default='gpu', type=str,  help='If npu, npu is used.')

    # distributed training
    parser.add_argument('--distributed', default=0, type=int, help="define gpu id")
    parser.add_argument('--device_lists', default='0', type=str, help="define gpu id")
    parser.add_argument('--device_num', default=1, type=int, help='Whether to use the multi- GPU/NPU.')
    parser.add_argument('--world-size', default=-1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='local_rank')
    parser.add_argument('--addr', default='127.0.0.1', type=str, help='addr')

    parser.add_argument('--use_prof', default=1, type=int, help='use_prof')
    parser.add_argument('--use_apex', default=0, type=int, help='use_apex')
    parser.add_argument('--opt_level', default='O2', type=str, help='Initial opt_level')
    parser.add_argument('--loss_scale', default='dynamic', help='Initial loss_scale')

    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=2, type=int, help='Number of total epochs to run')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--droupout_rate', default=0.9, type=float, help='Droupout rate')
    parser.add_argument('--lr_steps', default=[15, 25, 35, 40, 45], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--n_threads', default=0, type=int, help='Number of threads for multi-thread loading')

    parser.add_argument('--no_train', default=0, type=int, help='If true, training is not performed.')
    parser.add_argument('--no_val', default=0, type=int, help='If true, validation is not performed.')
    parser.add_argument('--test', default=1, type=int, help='If true, test is performed.')
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')

    parser.add_argument('--n_classes', default=101, type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_finetune_classes', default=101, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')

    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--modality', default='RGB', type=str, help='Modality of input data. RGB, Flow or RGBFlow')
    parser.add_argument('--dataset', default='ucf101', type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='random', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--mean_dataset', default='activitynet', type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--n_val_samples', default=1, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test', default='c', type=str, help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument('--no_softmax_in_test', action='store_true', help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--checkpoint', default=1, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='mobilenetv2', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--version', default=1.1, type=float, help='Version of the model')
    parser.add_argument('--groups', default=3, type=int, help='The number of groups at group convolutions at conv layers')
    parser.add_argument('--width_mult', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--manual_seed', default=2, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args
