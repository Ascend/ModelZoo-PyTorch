# Copyright 2021 Huawei Technologies Co., Ltd
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
import sys
import os
import stat
import time
import io
import gc
import logging
from bisect import bisect       # for lr_scheduler
from contextlib import redirect_stdout
from opt_loss import OptLoss
from mlperf_logger import configure_logger, log_start, log_end, log_event, set_seeds, get_rank, barrier
from mlperf_logging.mllog import constants
import torch
from torch.autograd import Variable
from base_model import Loss
from apex import amp
from ssd300 import SSD300
from master_params import create_flat_master
from parse_config import parse_args, validate_arguments, validate_group_bn
from data.build_pipeline import prebuild_pipeline, build_pipeline
from box_coder import dboxes300_coco, build_ssd300_coder
from async_evaluator import AsyncEvaluator
from eval import coco_eval
from apex.optimizers import NpuFusedSGD
from torch.nn.parallel import DistributedDataParallel
import numpy as np
# necessary pytorch imports
import torch.utils.data.distributed
import torch.distributed as dist
if torch.__version__ >= '1.8':
    import torch_npu
try:
    from torch_npu.utils.profiler import Profile
except Exception:
    print("Profile not in torch_npu.utils.profiler now.. Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass

# Apex imports
try:
    import apex_C
    import apex
    from apex.parallel.LARC import LARC
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import convert_network
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")


class Logger(object):
    logfile = ""

    def __init__(self, filename=""):
        self.logfile = filename
        self.terminal = sys.stdout
        return

    def write(self, message):
        self.terminal.write(message)
        if self.logfile != "":
            try:
                fd = os.open(self.logfile, os.O_RDWR|os.O_CREAT, stat.S_IRWXU)
                self.log = os.fdopen(fd, "a")
                self.log.write(message)
                self.log.close(fd)
            except Exception:
                pass

    def flush(self):
        pass


def print_message(rank, *print_args):
    if rank == 0:
        print(*print_args)

def load_checkpoint(model, checkpoint):
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)
    # remove proceeding 'module' from checkpoint
    saved_model = od["model"]
    for k in list(saved_model.keys()):
        if k.startswith('module.'):
            saved_model[k[7:]] = saved_model.pop(k)
    model.load_state_dict(saved_model)

def check_async_evals(args, evaluator, threshold):
    finished = 0
    # Note: only one rank does COCOEval, so we need to check there if we've
    # finished -- we'll broadcast that to a "finished" tensor to determine
    # if we should stop
    # Note2: ssd_print contains a barrier() call, implemented with all_reduce
    #        If we conditional on rank 0, then an ssd_print all_reduce matches with
    #        the finished all_reduce and all hell breaks loose.
    if args.rank == 0:
        for epoch, current_accuracy in evaluator.finished_tasks().items():
            # Note: Move to per-iter check
            # EVAL_START should be prior to the accuracy/score evaluation but adding the missing EVAL_START here for now
            log_start(key=constants.EVAL_START, metadata={'epoch_num' : epoch})
            log_event(key=constants.EVAL_ACCURACY,
                      value=current_accuracy,
                      metadata={'epoch_num' : epoch})
            log_end(key=constants.EVAL_STOP, metadata={'epoch_num' : epoch})
            if current_accuracy >= threshold:
                finished = 1

    # handle the non-distributed case -- don't need to bcast, just take local result
    if not args.distributed:
        return finished == 1

    # Now we know from all ranks if they're done - reduce result
    # Note: Already caught the non-distributed case above, can assume broadcast is available
    with torch.no_grad():
        finish_tensor = torch.tensor([finished], dtype=torch.int32, device=torch.device('npu'))
        torch.distributed.broadcast(finish_tensor, src=0)

        # >= 1 ranks has seen final accuracy
        if finish_tensor.item() >= 1:
            return True

    # Default case: No results, or no accuracte enough results
    return False

def lr_warmup(optim, warmup_iter, iter_num, epoch, base_lr, args):
    if iter_num < warmup_iter:
        warmup_step = base_lr / (warmup_iter * (2 ** args.warmup_factor))
        new_lr = base_lr - (warmup_iter - iter_num) * warmup_step

        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

def setup_distributed(args):
    # Setup multi-GPU if necessary

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    if args.distributed:
        torch.npu.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='hccl',
                                            world_size=int(os.environ['WORLD_SIZE']),
                                            rank=args.local_rank,
                                             )
    args.local_seed = set_seeds(args)
    # start timing here
    if args.distributed:
        args.N_gpu = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        args.N_gpu = 1
        args.rank = 0

    validate_group_bn(args.bn_group)

    return args

def train300_mlperf_coco(args):

    
    args = setup_distributed(args)

    if not args.distributed:
        torch.npu.set_device(args.device_id)
    
    # Build the model
    model_options = {
        'use_nhwc' : args.nhwc,
        'pad_input' : args.pad_input,
        'bn_group' : args.bn_group,
    }

    ssd300 = SSD300(args, args.num_classes, **model_options)
    if args.checkpoint is not None:
        load_checkpoint(ssd300, args.checkpoint)

    ssd300.train()
    ssd300.npu()
    dboxes = dboxes300_coco()
    loss_func = Loss(dboxes)
    loss_func.npu()

    # Create optimizer.  This must also be done after network_to_half.
    global_batch_size = (args.N_gpu * args.batch_size)
    log_event(key=constants.MODEL_BN_SPAN, value=args.bn_group*args.batch_size)
    log_event(key=constants.GLOBAL_BATCH_SIZE, value=global_batch_size)

    # mlperf only allows base_lr scaled by an integer
    base_lr = 2.5e-3
    requested_lr_multiplier = args.lr / base_lr
    adjusted_multiplier = max(1, round(requested_lr_multiplier * global_batch_size / 32))

    current_lr = base_lr * adjusted_multiplier
    current_momentum = 0.9
    current_weight_decay = args.wd
    static_loss_scale = args.loss_scale


    optim = apex.optimizers.NpuFusedSGD(ssd300.parameters(),
                                     lr=current_lr,
                                     momentum=current_momentum,
                                     weight_decay=current_weight_decay)
    ssd300, optim = amp.initialize(ssd300, optim, opt_level='O2', loss_scale=static_loss_scale,combine_grad=True)
    # Parallelize.  Need to do this after network_to_half.
    if args.distributed:
        if args.delay_allreduce:
            print_message(args.local_rank, "Delaying allreduces to the end of backward()")
        ssd300 = DistributedDataParallel(ssd300, device_ids=[args.local_rank])

    log_event(key=constants.OPT_BASE_LR, value=current_lr)
    log_event(key=constants.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=args.lr_decay_epochs)
    log_event(key=constants.OPT_LR_DECAY_STEPS, value=args.lr_decay_epochs)
    log_event(key=constants.OPT_WEIGHT_DECAY, value=current_weight_decay)
    if args.warmup is not None:
        log_event(key=constants.OPT_LR_WARMUP_STEPS, value=args.warmup)
        log_event(key=constants.OPT_LR_WARMUP_FACTOR, value=args.warmup_factor)

    # Model is completely finished -- need to create separate copies, preserve parameters across
    # them, and jit
    ssd300_eval = SSD300(args, args.num_classes, **model_options).npu()

    if args.use_fp16:
        convert_network(ssd300_eval, torch.half)

    # Get the existant state from the train model
    # * if we use distributed, then we want .module
    train_model = ssd300.module if args.distributed else ssd300
    ssd300_eval.load_state_dict(train_model.state_dict())
    ssd300_eval.eval()


    print_message(args.local_rank, "epoch", "nbatch", "loss")

    iter_num = args.iteration
    avg_loss = 0.0

    start_elapsed_time = time.time()
    last_printed_iter = args.iteration
    num_elapsed_samples = 0

    input_c = 4 if args.pad_input else 3
    example_shape = [args.batch_size, 300, 300, input_c] if args.nhwc else [args.batch_size, input_c, 300, 300]
    example_input = torch.randn(*example_shape).npu()

    if args.use_fp16:
        example_input = example_input.half()
    
    if args.jit:
        # DDP has some Python-side control flow.  If we JIT the entire DDP-wrapped module,
        # the resulting ScriptModule will elide this control flow, resulting in allreduce
        # hooks not being called.  If we're running distributed, we need to extract and JIT
        # the wrapped .module.
        # Replacing a DDP-ed ssd300 with a script_module might also cause the AccumulateGrad hooks
        # to go out of scope, and therefore silently disappear.
        module_to_jit = ssd300.module if args.distributed else ssd300
        if args.distributed:
            ssd300.module = torch.jit.trace(module_to_jit, example_input, check_trace=False)
        else:
            ssd300 = torch.jit.trace(module_to_jit, example_input, check_trace=False)
        # JIT the eval model too
        ssd300_eval = torch.jit.trace(ssd300_eval, example_input, check_trace=False)

    # do a dummy fprop & bprop to make sure cudnnFind etc. are timed here
    ploc, plabel = ssd300(example_input)

    # produce a single dummy "loss" to make things easier
    loss = ploc[0,0,0] + plabel[0,0,0]
    dloss = torch.randn_like(loss)
    # Cause cudnnFind for dgrad, wgrad to run
    loss.backward(dloss)

    encoder = build_ssd300_coder()

    evaluator = AsyncEvaluator(num_threads=1)

    log_end(key=constants.INIT_STOP)

    ##### END INIT

    # This is the first place we touch anything related to data
    ##### START DATA TOUCHING
    barrier()
    log_start(key=constants.RUN_START)
    barrier()

    train_pipe = prebuild_pipeline(args)
       
    train_loader, epoch_size = build_pipeline(args, training=True, pipe=train_pipe)
    if args.rank == 0:
        print("epoch size is: ", epoch_size, " images")

    val_loader, inv_map, cocoGt = build_pipeline(args, training=False)
    if args.profile_gc_off:
        gc.disable()
        gc.collect()

    ##### END DATA TOUCHING
    i_eval = 0
    block_start_epoch = 1
    log_start(key=constants.BLOCK_START,
              metadata={'first_epoch_num': block_start_epoch,
                        'epoch_count': args.evaluation[i_eval]})
    for epoch in range(args.epochs):
        optim.zero_grad()
        

        if epoch in args.evaluation:
            # Get the existant state from the train model
            # * if we use distributed, then we want .module
            train_model = ssd300.module if args.distributed else ssd300

            if args.distributed and args.allreduce_running_stats:
                if args.rank == 0:
                    print("averaging bn running means and vars")
                # make sure every node has the same running bn stats before
                # using them to evaluate, or saving the model for inference
                world_size = float(torch.distributed.get_world_size())
                for bn_name, bn_buf in train_model.named_buffers(recurse=True):
                    if ('running_mean' in bn_name) or ('running_var' in bn_name):
                        torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                        bn_buf /= world_size

            if args.rank == 0:
                if args.save:
                    print("saving model...")
                    if not os.path.isdir('./models'):
                        os.mkdir('./models')
                    torch.save({"model" : ssd300.state_dict()}, "./models/iter_{}.pt".format(iter_num))
            
            ssd300_eval.load_state_dict(train_model.state_dict())
            # Note: No longer returns, evaluation is abstracted away inside evaluator
            coco_eval(args,
                      ssd300_eval,
                      val_loader,
                      cocoGt,
                      encoder,
                      inv_map,
                      epoch,
                      iter_num,
                      evaluator=evaluator)
            log_end(key=constants.BLOCK_STOP, metadata={'first_epoch_num': block_start_epoch})
            if epoch != max(args.evaluation):
                i_eval += 1
                block_start_epoch = epoch + 1
                log_start(key=constants.BLOCK_START,
                          metadata={'first_epoch_num': block_start_epoch,
                                    'epoch_count': (args.evaluation[i_eval] -
                                                    args.evaluation[i_eval - 1])})

        if epoch in args.lr_decay_epochs:
            current_lr *= args.lr_decay_factor
            print_message(args.rank, "lr decay step #" + str(bisect(args.lr_decay_epochs, epoch)))
            for param_group in optim.param_groups:
                param_group['lr'] = current_lr

        log_start(key=constants.EPOCH_START,
                  metadata={'epoch_num': epoch + 1,
                            'current_iter_nufm': iter_num})

        profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                          profile_type=os.getenv('PROFILE_TYPE'))

        for i, data in enumerate(train_loader):
            (img, bbox, label, _) = data
            img = img.npu()
            bbox = bbox.npu()
            label = label.npu()
            if args.profile_start is not None and iter_num == args.profile_start:
                torch.npu.profiler.start()
                torch.npu.synchronize()
                if args.profile_nvtx:
                    torch.autograd._enable_profiler(torch.autograd.ProfilerState.NVTX)

            if args.profile is not None and iter_num == args.profile:
                if args.profile_start is not None and iter_num >=args.profile_start:
                    # we turned npu and nvtx profiling on, better turn it off too
                    if args.profile_nvtx:
                        torch.autograd._disable_profiler()
                    torch.npu.profiler.stop()
                sys.exit()

            if args.warmup is not None:
                lr_warmup(optim, args.warmup, iter_num, epoch, current_lr, args)

            if (img is None) or (bbox is None) or (label is None):
                print("No labels in batch")
                continue

            profile.start()
            ploc, plabel = ssd300(img)
            ploc, plabel = ploc.float(), plabel.float()

            N = img.shape[0]
            bbox.requires_grad = False
            label.requires_grad = False
            # reshape (N*8732X4 -> Nx8732x4) and transpose (Nx8732x4 -> Nx4x8732)
            bbox = bbox.view(N, -1, 4).transpose(1,2).contiguous()
            # reshape (N*8732 -> Nx8732) and cast to Long
            label = label.view(N, -1).long()
            loss = loss_func(ploc, plabel, bbox, label)

            if np.isfinite(loss.item()):
                avg_loss = 0.999*avg_loss + 0.001*loss.item()
            else:
                print("model exploded (corrupted by Inf or Nan)")
                sys.exit()

            num_elapsed_samples += N
            # if args.rank == 0 and iter_num % args.print_interval == 0:
            if args.rank == 0 and iter_num % args.print_interval == 0:
                end_elapsed_time = time.time()
                elapsed_time = end_elapsed_time - start_elapsed_time

                avg_samples_per_sec = num_elapsed_samples * args.N_gpu / elapsed_time

                print("Epoch:{:4d}, Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f},\
                      avg. samples / sec: {:.2f}".format(epoch, iter_num, loss.item(),\
                      avg_loss, avg_samples_per_sec), end="\n")

                last_printed_iter = iter_num
                start_elapsed_time = time.time()
                num_elapsed_samples = 0

            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()


            optim.step()

            # Likely a decent skew here, let's take this opportunity to set the
            # gradients to None.  After DALI integration, playing with the
            # placement of this is worth trying.

            optim.zero_grad()
            profile.end()

            # Don't check every iteration due to cost of broadcast
            if iter_num % 20 == 0:
                finished = check_async_evals(args, evaluator, args.threshold)

                if finished:
                    return True

            iter_num += 1

        log_end(key=constants.EPOCH_STOP, metadata={'epoch_num': epoch + 1})

    return False

def main():
    configure_logger(constants.SSD)
    log_start(key=constants.INIT_START, log_all_ranks=True)
    args = parse_args()
    sys.stdout = Logger("test/output/%s/%s_%s.log"%(args.device_id,args.tag,args.device_id))
    # 1p
    sys.stderr = Logger("test/output/%s/%s_%s.log"%(args.device_id,args.tag,args.device_id))
    if args.local_rank == 0:
        print(args)

    # make sure the epoch lists are in sorted order
    args.evaluation.sort()
    args.lr_decay_epochs.sort()

    validate_arguments(args)

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = not args.profile_cudnn_get

    success = train300_mlperf_coco(args)
    status = 'success' if success else 'aborted'

    # end timing here
    log_end(key=constants.RUN_STOP, metadata={'status': status})


if __name__ == "__main__":
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = '''BNTrainingReduce,BNTrainingReduceGrad,
                                             BNTrainingUpdate,BNTrainingUpdateGrad'''
    torch.npu.set_option(option)
    main()
