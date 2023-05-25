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

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.distributed as dist
import numpy as np
import hashlib
from common.generators import UnchunkedGenerator
from common.loss import *
import logging

logger = logging.getLogger(__name__)

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value


def fetch(subjects, keypoints, dataset, args, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
    
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    

    return out_camera_params, out_poses_3d, out_poses_2d


def evaluate(test_generator, model_pos, model_traj, joints_left, joints_right, gpu, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    loc = f'npu:{gpu}'
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        else:
            model_traj.eval()
        N = 0
        # count_idx = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            # print(f"inputs_2d.shape:{inputs_2d.shape} for {count_idx} in rank {dist.get_rank()}")
            # count_idx += 1
            if torch.npu.is_available():
                inputs_2d = inputs_2d.to(loc)

            # Positional model
            if not use_trajectory_model:
                predicted_3d_pos = model_pos(inputs_2d)
            else:
                predicted_3d_pos = model_traj(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    # due to certain mechanism in npu, cast type before and after assignment
                    predicted_3d_pos = torch_npu.npu_format_cast(predicted_3d_pos, 0)
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                    predicted_3d_pos = torch_npu.npu_format_cast(predicted_3d_pos, 3)
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
                
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.npu.is_available():
                inputs_3d = inputs_3d.to(loc)
            inputs_3d[:, :, 0] = 0    
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)
         
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
    # assert 1==2        
    # if action is None:
    #     logger.info('----------')
    # else:
    #     logger.info('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    # print('Test time augmentation:', test_generator.augment_enabled())
    # print('Protocol #1 Error (MPJPE):', e1, 'mm')
    # print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    # print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    # print('Velocity Error (MPJVE):', ev, 'mm')
    # print('----------')

    e1 = torch.tensor(e1, dtype=error.dtype, device=error.device)
    e2 = torch.tensor(e2, dtype=error.dtype, device=error.device)
    e3 = torch.tensor(e3, dtype=error.dtype, device=error.device)
    ev = torch.tensor(ev, dtype=error.dtype, device=error.device)

    return e1, e2, e3, ev


def fetch_actions(args, actions, keypoints, dataset):
    out_poses_3d = []
    out_poses_2d = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)): # Iterate across cameras
            out_poses_3d.append(poses_3d[i])

    stride = args.downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    
    return out_poses_3d, out_poses_2d


def run_evaluation(args, actions, model_pos, model_traj, keypoints, dataset, pad, causal_shift, kps_left, kps_right, joints_left, joints_right, action_filter=None):
    errors_p1 = []
    errors_p2 = []
    errors_p3 = []
    errors_vel = []

    for action_key in actions.keys():
        if action_filter is not None:
            found = False
            for a in action_filter:
                if action_key.startswith(a):
                    found = True
                    break
            if not found:
                continue

        poses_act, poses_2d_act = fetch_actions(args, actions[action_key], keypoints, dataset)
        gen = UnchunkedGenerator(args, None, poses_act, poses_2d_act,
                                pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        e1, e2, e3, ev = evaluate(gen, model_pos, model_traj, joints_left, joints_right, args.gpu, action_key)

        # e1_list = [torch.zeros_like(e1) for _ in range(args.num_gpus)]
        # dist.all_gather(e1_list,e1)
        # e1 = torch.tensor(e1_list)
        # e1 = e1.mean()

        # e2_list = [torch.zeros_like(e2) for _ in range(args.num_gpus)]
        # dist.all_gather(e2_list,e2)
        # e2 = torch.tensor(e2_list)
        # e2 = e2.mean()

        # e3_list = [torch.zeros_like(e3) for _ in range(args.num_gpus)]
        # dist.all_gather(e3_list,e3)
        # e3 = torch.tensor(e3_list)
        # e3 = e3.mean()

        # ev_list = [torch.zeros_like(ev) for _ in range(args.num_gpus)]
        # dist.all_gather(ev_list,ev)
        # ev = torch.tensor(ev_list)
        # ev = ev.mean()

        dist.all_reduce(e1)
        e1 = e1/args.num_gpus
        e1 = e1.cpu().numpy()
        dist.all_reduce(e2)
        e2 = e2/args.num_gpus
        e2 = e2.cpu().numpy()
        dist.all_reduce(e3)
        e3 = e3/args.num_gpus
        e3 = e3.cpu().numpy()
        dist.all_reduce(ev)
        ev = ev/args.num_gpus
        ev = ev.cpu().numpy()

        errors_p1.append(e1)
        errors_p2.append(e2)
        errors_p3.append(e3)
        errors_vel.append(ev)

    # print(f"errors_p1:{errors_p1} for rank {args.rank}")
    # emp1 = torch.tensor(errors_p1,device=e1.device)
    # emp1 = emp1.mean()
    # print(f"emp1:{emp1} for rank {args.rank}")
    # emp1 = emp1.mean()
    # emp2 = torch.tensor(errors_p2,device=e1.device)
    # emp2 = emp2.mean()
    # emp3 = torch.tensor(errors_p3,device=e1.device)
    # emp3 = emp3.mean()
    # emv = torch.tensor(errors_vel,device=e1.device)
    # emv = emv.mean()

    if args.rank % args.num_gpus == 0:
        logger.info(f'Protocol #1   (MPJPE) action-wise average:{round(np.mean(errors_p1), 1)}mm')
        logger.info(f'Protocol #2 (P-MPJPE) action-wise average:{round(np.mean(errors_p2), 1)}mm')
        logger.info(f'Protocol #3 (N-MPJPE) action-wise average:{round(np.mean(errors_p3), 1)}mm')
        logger.info(f'Velocity      (MPJVE) action-wise average:{round(np.mean(errors_vel), 2)}mm')

