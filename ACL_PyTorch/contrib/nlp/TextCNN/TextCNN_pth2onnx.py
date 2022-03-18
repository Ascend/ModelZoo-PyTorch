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

import sys, argparse, pickle as pkl, torch

sys.path.append(r'./Chinese-Text-Classification-Pytorch')
from models import TextCNN

parser = argparse.ArgumentParser(description='TextCNN_pth2onnx.py')
parser.add_argument('--weight_path', required=True, help='Path to model weight file, abs path recommended.')
parser.add_argument('--dataset', default='./Chinese-Text-Classification-Pytorch/THUCNews',
                    help="""Dataset path, train: $dataset/data/train.txt, dev: $dataset/data/dev.txt, \n
                    test: $dataset/data/text.txt, classes list: $dataset/data/class.txt, \n 
                    vocab: $dataset/data/vocab.pkl, embedding file should be in $dataset/data/""")
parser.add_argument('--embedding', default='embedding_SougouNews.npz',
                    help="embedding file of $dataset/data/")
parser.add_argument('--onnx_path', required=True, help='Path to save onnx weights.')
args = parser.parse_args()


def main():
    config = TextCNN.Config(args.dataset, args.embedding)
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    config.n_vocab = len(vocab)

    model = TextCNN.Model(config)
    model.load_state_dict(torch.load(args.weight_path, map_location=config.device))
    model.eval()
    input_names = ['sentence']
    output_names = ['class']
    dynamic_axes = {'sentence': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randint(100, (1, 32))
    torch.onnx.export(model, dummy_input, args.onnx_path, input_names=input_names, verbose=True,
                      output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)

if __name__ == '__main__':
    """
    Usage Example:
    python TextCNN_pth2onnx.py \
    --weight_path ./Chinese-Text-Classification-Pytorch/THUCNews/saved_dict/TextCNN.ckpt \
    --onnx_path ./Chinese-Text-Classification-Pytorch/THUCNews/saved_dict/TextCNN_onnx.onnx
    """
    main()
