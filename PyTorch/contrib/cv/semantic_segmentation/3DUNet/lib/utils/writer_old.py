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


# TODO create class Writer


def write_score(writer, iter, loss_dice, dice_coeff, per_ch_score, mode="Train/"):
    writer.add_scalar(mode + 'loss_dice', loss_dice, iter)
    writer.add_scalar(mode + 'dice_coeff', dice_coeff, iter)
    writer.add_scalar(mode + 'air', per_ch_score[0], iter)
    writer.add_scalar(mode + 'csf', per_ch_score[1], iter)
    writer.add_scalar(mode + 'gm', per_ch_score[2], iter)
    writer.add_scalar(mode + 'wm', per_ch_score[3], iter)


def write_train_val_score(writer, epoch, train_stats, val_stats):
    writer.add_scalars('Loss', {'train': train_stats[0],
                                'val': val_stats[0],
                                }, epoch)
    writer.add_scalars('Coeff', {'train': train_stats[1],
                                 'val': val_stats[1],
                                 }, epoch)

    writer.add_scalars('Air', {'train': train_stats[2],
                               'val': val_stats[2],
                               }, epoch)

    writer.add_scalars('CSF', {'train': train_stats[3],
                               'val': val_stats[3],
                               }, epoch)
    writer.add_scalars('GM', {'train': train_stats[4],
                              'val': val_stats[4],
                              }, epoch)
    writer.add_scalars('WM', {'train': train_stats[5],
                              'val': val_stats[5],
                              }, epoch)
    return
