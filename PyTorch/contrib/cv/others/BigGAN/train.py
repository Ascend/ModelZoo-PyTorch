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
""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import time

import torch
import torch.distributed as dist

# Import my stuff
import inception_utils
import train_fns
import utils


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def get_device_name(device_type, device_order):
    if device_type == 'npu':
        device_name = 'npu:{}'.format(device_order)
    else:
        device_name = 'cuda:{}'.format(device_order)

    return device_name


def profiling(data_loader, G, D, train, config):
    print("profiling mode ...")
    G.train()
    D.train()
    for i, (x, y) in enumerate(data_loader):
        # Make sure G and D are in training mode, just in case they got set to eval
        # For D, which typically doesn't have BN, this shouldn't matter much.

        if config['D_fp16']:
            x, y = x.to(config['loc']).half(), y.to(config['loc'])
        else:
            x, y = x.to(config['loc']), y.to(config['loc'])

        if i < 5:
            print("iter: ", i)
            train(x, y)
        else:
            if config['device'] == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    train(x, y)
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    train(x, y)
            break
    prof.export_chrome_trace("%s.prof" % config['device'])


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(gpu, ngpus_per_node, config):
    config['gpu'] = config['process_device_map'][gpu]
    if config['distributed']:
        print("use distributed training... gpu:", config['gpu'])
        if config['device'] == 'npu':
            dist.init_process_group(backend=config['dist_backend'],
                                    world_size=config['world_size'],
                                    rank=config['rank'])
        else:
            dist.init_process_group(backend=config['dist_backend'],
                                    init_method=config['dist_url'],
                                    world_size=config['world_size'],
                                    rank=config['rank'])
    print('rank: {} / {}'.format(config['rank'], config['world_size']))
    # init device
    device_loc = get_device_name(config['device'], config['gpu'])
    config['loc'] = device_loc
    # set device
    print('set_device ', device_loc)
    if config['device'] == 'npu':
        torch.npu.set_device(device_loc)
    else:
        torch.cuda.set_device(config['gpu'])
    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    # Next, build the model
    G = model.Generator(**config).to(device_loc)
    D = model.Discriminator(**config).to(device_loc)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init': True, 'no_optim': True}).to(device_loc)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    if config['distributed']:
        config['batch_size'] = int(config['batch_size'] / config['world_size'])
        config['num_workers'] = int((config['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)

    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?
    GD = model.G_D(G, D)

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...device:', device_loc)
        utils.load_weights(G, D, state_dict,
                           config['weights_root'], config['experiment_name'],
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)

    if config['distributed']:
        GD = torch.nn.parallel.DistributedDataParallel(GD, device_ids=[config['gpu']], find_unused_parameters=True)

    if not config['distributed'] or (config['distributed'] and config['gpu'] == config['process_device_map'][0]):
        # Prepare loggers for stats; metrics holds test metrics,
        # lmetrics holds any desired training metrics.
        test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                                  config['experiment_name'])
        train_metrics_fname = '%s/%s' % (config['logs_root'], config['experiment_name'])
        print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
        test_log = utils.MetricsLogger(test_metrics_fname,
                                       reinitialize=(not config['resume']))
        print('Training Metrics will be saved to {}'.format(train_metrics_fname))
        train_log = utils.MyLogger(train_metrics_fname,
                                   reinitialize=(not config['resume']),
                                   logstyle=config['logstyle'])
    else:
        test_log = None
        train_log = None
    if not config['distributed'] or (config['distributed'] and config['gpu'] == config['process_device_map'][0]):
        # Write metadata
        utils.write_metadata(config['logs_root'], config['experiment_name'], config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])
    loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                        'start_itr': state_dict['itr']})
    if config['distributed']:
        train_sampler = loaders[0]
        loader = loaders[1]
    else:
        loader = loaders[0]

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device_loc, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device_loc,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                                ema, state_dict, config)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()

    if config['prof']:
        profiling(loader, G, D, train, config)
        return

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations.
    start_time = time.time()
    total = config['num_epochs'] * len(loader)
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        if config['distributed']:
            train_sampler.set_epoch(epoch)
        batch_time = utils.AverageMeter('Time', ':6.3f')
        data_time = utils.AverageMeter('Data', ':6.3f')
        end = time.time()
        for i, (x, y) in enumerate(loader):
            data_time.update(time.time() - end)
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device_loc).half(), y.to(device_loc)
            else:
                x, y = x.to(device_loc), y.to(device_loc)
            metrics = train(x, y)
            # measure elapsed time
            cost_time = time.time() - end
            batch_time.update(cost_time)
            end = time.time()

            metrics['data_val'] = data_time.val
            metrics['data_avg'] = data_time.avg
            metrics['batch_val'] = batch_time.val
            metrics['batch_avg'] = batch_time.avg
            metrics['FPS'] = D_batch_size * config['world_size'] / batch_time.avg if batch_time.avg else 0

            if not config['distributed'] or (
                    config['distributed'] and config['gpu'] == config['process_device_map'][0]):
                train_log.log(itr=int(state_dict['itr']), epoch=epoch, **metrics)

                # Every sv_log_interval, log singular values
                if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
                    train_log.log(itr=int(state_dict['itr']), epoch=epoch,
                                  **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

                # If using my progbar, print metrics.
                if config['pbar'] == 'mine':
                    print(', '.join(
                        ["Epoch: %d" % epoch,
                         'itr/total: %d/%d' % (state_dict['itr'], total),
                         "time: %d:%02d" % tuple(divmod(time.time() - start_time, 60))]
                        + ['%s : %+4.3f' % (key, metrics[key]) for key in metrics]), end=' ')
                    print()

                # Save weights and copies as configured at specified interval
                if not (state_dict['itr'] % config['save_every']):
                    if config['G_eval_mode']:
                        G.eval()
                        if config['ema']:
                            G_ema.eval()
                    train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                              state_dict, config, config['experiment_name'], device=device_loc)

            if config['cann_prof']:
                return

            if 0 < config['stop_iter'] == state_dict['itr']:
                return
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    config['process_device_map'] = device_id_to_process_device_map(config['device_list'])

    os.environ['MASTER_ADDR'] = config['addr']
    os.environ['MASTER_PORT'] = '29688'
    # Seed RNG
    utils.seed_rng(config['seed'])
    if config['device'] == 'npu':
        ngpus_per_node = len(config['process_device_map'])
    else:
        if config['gpu'] is None:
            ngpus_per_node = len(config['process_device_map'])
        else:
            ngpus_per_node = 1
    config['world_size'] = ngpus_per_node * config['world_size']
    config['distributed'] = config['world_size'] > 1
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    config['experiment_name'] = experiment_name
    run(config['rank'], ngpus_per_node, config)


if __name__ == '__main__':
    main()
