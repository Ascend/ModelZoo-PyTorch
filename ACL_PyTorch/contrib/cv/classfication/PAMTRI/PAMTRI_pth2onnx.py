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


from torchreid import models
from torchreid.data_manager import DatasetManager
import argparse
import torch
import sys
sys.path.append('./PAMTRI/MultiTaskNet')


def main(args):
    dataset = DatasetManager(dataset_dir=args.dataset,
                             root=args.root,
                             verbose=False)
    model = models.init_model(name=args.arch,
                              num_vids=dataset.num_train_vids,
                              num_vcolors=dataset.num_train_vcolors,
                              num_vtypes=dataset.num_train_vtypes,
                              keyptaware=args.keyptaware,
                              heatmapaware=args.heatmapaware,
                              segmentaware=args.segmentaware,
                              multitask=args.multitask)
    checkpoint = torch.load(args.load_weights,
                            map_location=torch.device('cpu'))
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items()
                     if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    dummy_input = torch.randn(1, 3, args.width, args.height)
    output_names = ['output1', 'output2', 'output3', 'output4']
    dynamic_axes = {'input': {0: 'batch_size'},
                    'output1': {0: 'batch_size'},
                    'output2': {0: 'batch_size'},
                    'output3': {0: 'batch_size'},
                    'output4': {0: 'batch_size'}
                    }
    torch.onnx.export(model, dummy_input, args.output_path,
                      verbose=False, input_names=['input'],
                      output_names=output_names, opset_version=11,
                      dynamic_axes=dynamic_axes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Export to onnx')
    # Datasets
    parser.add_argument('--root', type=str, default='./PAMTRI/MultiTaskNet/data',
                        help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='veri',
                        help="name of the dataset")
    # Architecture
    parser.add_argument('-a', '--arch', type=str,
                        default='densenet121', choices=models.get_names())
    parser.add_argument('--keyptaware', action='store_true',
                        help="embed keypoints to deep features")
    parser.add_argument('--heatmapaware', action='store_true',
                        help="embed heatmaps to images")
    parser.add_argument('--segmentaware', action='store_true',
                        help="embed segments to images")
    parser.add_argument('--load-weights', type=str,
                        help="load pretrained weights but ignores layers that don't match in size")
    parser.add_argument('--multitask', action='store_true',
                        help="use multi-task learning")
    parser.add_argument('--output_path', type=str,
                        help="output path")
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image (default: 256)")
    parser.add_argument('--width', type=int, default=256,
                        help="width of an image (default: 256)")
    args = parser.parse_args()

    main(args)
