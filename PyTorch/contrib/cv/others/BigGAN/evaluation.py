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
import functools
import torch
import inception_utils
import utils
from train import get_device_name


def evaluation(config):
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    # Seed RNG
    utils.seed_rng(config['seed'])

    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True

    # init device
    device_loc = get_device_name(config['device'], config['gpu'])
    config['loc'] = device_loc
    # set device
    print('set_device ', device_loc)
    if config['device'] == 'npu':
        torch.npu.set_device(device_loc)
    else:
        torch.cuda.set_device(config['gpu'])

    # model
    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    # Next, build the model
    G = model.Generator(**config).to(device_loc)

    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...gpu id : ', config['gpu'])
        utils.load_weights(G, None, state_dict,
                           config['weights_root'], config['experiment_name'],
                           config['load_weights'] if config['load_weights'] else None,
                           None, root=config['weights_path'], load_optim=False)
        print("load weights ok")

    # prepare input
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device_loc)
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)
    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'],
                                                                      config['no_fid'], config['loc'],
                                                                      config['use_fp16'], config['opt_level'])
    if config['G_eval_mode']:
        G.eval()
    IS_mean, IS_std, FID = get_inception_metrics(sample, config['num_inception_images'], num_splits=10)
    log_string = "IS_mean: {:.5f}, IS_std: {:.5f}, FID: {:.5f}".format(IS_mean, IS_std, FID)
    print(log_string)
    with open("evaluation_log.log", "a+") as f:
        f.write("itr: {} , {:s}\n".format(state_dict['itr'], log_string))


if __name__ == "__main__":
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    evaluation(config)
