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


import argparse
import torch
import torch.onnx
from timm.models import create_model
import onnx
import tnt
from onnx import TensorProto


config_parser = parser = argparse.ArgumentParser(
    description="TNT Baseline Inference")
parser.add_argument('--model', default='tnt_s_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument("-c", "--config", default="", type=str, metavar="FILE",
                    help="YAML config file specifying default arguments")
parser.add_argument("--num-classes", type=int, default=1000, metavar="N",
                    help="number of label classes (default: 1000)")
parser.add_argument("--no-prefetcher", action="store_true", default=False,
                    help="disable fast prefetcher")
parser.add_argument("--pretrain_path",
                    default="./tnt_s_81.5.pth.tar", type=str)
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size of output onnx')


error_list_bs1 = ['Expand_199', 'Expand_207', 'Expand_443', 'Expand_451', 'Expand_687', 'Expand_695',
                  'Expand_931', 'Expand_939', 'Expand_1175', 'Expand_1183', 'Expand_1419', 'Expand_1427',
                  'Expand_1663', 'Expand_1671', 'Expand_1907', 'Expand_1915', 'Expand_2151', 'Expand_2159',
                  'Expand_2395', 'Expand_2403', 'Expand_2639', 'Expand_2647', 'Expand_2883', 'Expand_2891']

error_list_bs16 = ['Expand_190', 'Expand_198', 'Expand_425', 'Expand_433', 'Expand_660', 'Expand_668',
                   'Expand_895', 'Expand_903', 'Expand_1130', 'Expand_1138', 'Expand_1365', 'Expand_1373',
                   'Expand_1600', 'Expand_1608', 'Expand_1835', 'Expand_1843', 'Expand_2070', 'Expand_2078',
                   'Expand_2305', 'Expand_2313', 'Expand_2540', 'Expand_2548', 'Expand_2775', 'Expand_2783']

ERROR_DICT = {1: error_list_bs1, 16: error_list_bs16}


def sel_onnx(model, err):
    """select onnx node

    Args:
        model (onnx model): the model
        err (int): index of error node

    Returns:
        int: the node that gives the output
    """
    graph = model.graph
    nodes = graph.node
    for i in range(len(nodes)):
        if nodes[i].output[0] == err:
            return i


def onnx_modify(model, err):
    """modify onnx model

    Args:
        model (onnx_model): model to be modified
        err (int): index of error node
    """
    graph = model.graph
    nodes = graph.node
    for i in range(len(nodes)):
        if nodes[i].name == err:
            new_node = onnx.helper.make_node('Cast',
                                             name=err + '_input_cast',
                                             inputs=['input'],
                                             outputs=['input_for_' + err],
                                             to=getattr(TensorProto, 'INT32'),
                                             )
            #nodes[i].input[0] = new_node.output[0]
            pre_node_index = sel_onnx(model, nodes[i].input[0])
            new_node.input[0] = nodes[pre_node_index].output[0]
            nodes[i].input[0] = new_node.output[0]
            model.graph.node.insert(i, new_node)
            new_node = onnx.helper.make_node('Cast',
                                             name=err + '_output_cast',
                                             inputs=['input'],
                                             outputs=['output_for_' + err],
                                             to=getattr(TensorProto, 'INT64'),
                                             )
            new_node.input[0] = nodes[i + 1].output[0]
            for j in range(len(nodes)):
                if not nodes[j].input:
                    continue
                if nodes[i + 1].output[0] == nodes[j].input[0]:
                    nodes[j].input[0] = new_node.output[0]
            model.graph.node.insert(i + 2, new_node)
            return


def pth2onnx(model, args):
    """convert model to onnx model

    Args:
        model (model): the model
        args (args): args

    Raises:
        ValueError: pretrained weights not found
    """
    if args.pretrain_path is None:
        raise ValueError("pretrain path required for onnx")
    print("Loading:", args.pretrain_path)
    state_dict = torch.load(args.pretrain_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    print("Pretrain weights loaded")
    model.eval()
    input_names = ["inner_tokens"]
    output_names = ["class"]
    dynamic_axes = {"inner_tokens": {0: "-1"}, "class": {0: "-1"}}
    dummy_input = torch.randn(args.batch_size, 196, 16, 24)
    torch.onnx.export(model, dummy_input, args.model + "_bs{}.onnx".format(args.batch_size), input_names=input_names,
                      dynamic_axes=dynamic_axes, output_names=output_names, opset_version=11, verbose=True)


def to_numpy(tensor):
    """convert tensor to ndarray

    Args:
        tensor (tensor): tensor

    Returns:
        ndarray: converted tensor
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    """Main function
    """
    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher
    args.distributed = False

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=1000,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint)
    pth2onnx(model, args)
    onnx_model = onnx.load(
        'tnt_s_patch16_224_bs{}.onnx'.format(args.batch_size))
    if args.batch_size == 1:
        error_list = ERROR_DICT[1]
    else:
        error_list = ERROR_DICT[16]
    for err in error_list:
        onnx_modify(onnx_model, err)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, 'tnt_s_patch16_224_bs{}_cast.onnx'.format(
        args.batch_size))


if __name__ == "__main__":
    main()
