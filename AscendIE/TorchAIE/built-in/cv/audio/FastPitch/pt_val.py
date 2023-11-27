# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import argparse
import models
import sys
import copy
import numpy as np
import torch
import torch_aie

from pathlib import Path
from torch_aie import _enums
from torch.utils.data import dataloader
from torch.nn.utils.rnn import pad_sequence

from model_pt import forward_nms_script
from common.text.text_processing import TextProcessing
from waveglow import model as glow

sys.modules['glow'] = glow
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP

class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def collate_fn(batch):
        img = batch  # transposed
        return img


def load_model_from_ckpt(checkpoint_path, ema, model):

    checkpoint_data = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    status = ''

    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')

        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k,v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')

    return model


def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):

    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = models.get_model_config(model_name, model_args)

    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        model = load_model_from_ckpt(checkpoint, ema, model)

    if model_name == "WaveGlow":
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()

        model = model.remove_weightnorm(model)

    if amp:
        model.half()
    model.eval()
    return model.to(device)


def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c:f for c, f in zip(columns, fields)}


def prepare_input_sequence(fields, device, symbol_set, text_cleaners,
                           batch_size=128, dataset=None, load_mels=False,
                           load_pitch=False, p_arpabet=0.0):
    tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)

    fields['text'] = [torch.LongTensor(tp.encode_text(text))
                      for text in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    for t in fields['text']:
        print(tp.sequence_to_text(t.numpy()))

    if load_mels:
        assert 'mel' in fields
        fields['mel'] = [
            torch.load(Path(dataset, fields['mel'][i])).t() for i in order]
        fields['mel_lens'] = torch.LongTensor([t.size(0) for t in fields['mel']])

    if load_pitch:
        assert 'pitch' in fields
        fields['pitch'] = [
            torch.load(Path(dataset, fields['pitch'][i])) for i in order]
        fields['pitch_lens'] = torch.LongTensor([t.size(0) for t in fields['pitch']])

    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == 'mel' and load_mels:
                batch[f] = pad_sequence(batch[f], batch_first=True).permute(0, 2, 1)
            elif f == 'pitch' and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        size_0 = batch['text'].size(0)
        # print("111", batch, type(batch), type(batch['mel']))
        while(size_0 < batch_size):
            # print(batch)
            mel_li = list(batch['mel'])
            mel_li.append(mel_li[-1])
            batch['mel'] = tuple(mel_li)
            batch['output'].append(batch['output'][-1])
            text_li = batch['text'].numpy().tolist()
            text_li.append(copy.deepcopy(text_li[-1]))
            batch['text'] = torch.tensor(text_li)
            lens_li = batch['text_lens'].numpy().tolist()
            lens_li.append(lens_li[-1])
            batch['text_lens'] = torch.tensor(lens_li)
            # print(batch)
            size_0 += 1
        batches.append(batch)

    return batches

def main_datasets(opt, unk_args):
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """

    torch.backends.cudnn.benchmark = opt.cudnn_benchmark

    device = torch.device('cpu')

    if opt.fastpitch != 'SKIP':
        generator = load_and_setup_model(
            'FastPitch', parser, opt.fastpitch, opt.amp, device,
            unk_args=unk_args, forward_is_infer=True, ema=opt.ema,
            jitable=opt.torchscript)

        if opt.torchscript:
            generator = torch.jit.script(generator)
    else:
        generator = None

    fields = load_fields(opt.input)
    batches = prepare_input_sequence(
        fields, device, opt.symbol_set, opt.text_cleaners, 1,
        opt.dataset_path, load_mels=(generator is None), p_arpabet=opt.p_arpabet)

    datasets = []

    multi = opt.multi
    for n in range(multi):
        print(n / multi)
        for i, b in enumerate(batches):
            with torch.no_grad():
                # print(i)

                text_padded = torch.LongTensor(1, 200)
                text_padded.zero_()
                # print(b['text'])
                text_padded[:, :b['text'].size(1)] = b['text']
                datasets.append(text_padded)
    return datasets


def create_dataloader(opt, unk_args):
    dataset = main_datasets(opt, unk_args)
    loader =  InfiniteDataLoader  # only DataLoader allows for attribute updates
    nw = min([os.cpu_count() // WORLD_SIZE, opt.batch_size if opt.batch_size > 1 else 0, opt.n_workers])  # number of workers
    return loader(dataset,
                  batch_size=opt.batch_size,
                  shuffle=False,
                  num_workers=nw,
                  sampler=None,
                  pin_memory=True,
                  collate_fn=collate_fn)

def main(opt, unk_args):
    # load model
    model = torch.jit.load(opt.model)
    torch_aie.set_device(opt.device_id)
    if opt.need_compile:
        inputs = []
        inputs.append(torch_aie.Input((opt.batch_size, 200)))
        model = torch_aie.compile(
            model,
            inputs=inputs,
            precision_policy=_enums.PrecisionPolicy.FP16,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version=opt.soc_version,
            optimization_level=0)

    dataloader = create_dataloader(opt, unk_args)
    pred_results = forward_nms_script(model, dataloader, opt.batch_size, opt.device_id)
    if opt.multi == 1 and opt.batch_size == 1:
        result_path = "result/"
        if(os.path.exists(result_path) == False):
            os.makedirs(result_path)
        for index, res in enumerate(pred_results):
            # print(res[0].shape)
            for i, r in enumerate(res[0]):
                result_fname = 'data' + str(index * opt.batch_size + i) + '_0.bin'
                np.array(r.numpy().tofile(os.path.join(result_path, result_fname)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastPitch offline model inference.')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3', help='soc version')
    parser.add_argument('--model', type=str, default="fastpitch_torch_aie_bs4.pt", help='model path')
    parser.add_argument('--need_compile', action="store_true", help='if the loaded model needs to be compiled or not')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--multi', type=int, default=1, help='multiples of dataset replication for enough infer loop. if multi != 1, the pred result will not be stored.')
    parser.add_argument('--img_size', nargs='+', type=int, default=96, help='inference size (pixels)')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('-d', '--dataset_path', type=str,
                        default='./LJSpeech-1.1', help='Path to dataset')
    parser.add_argument('--n_speakers', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--symbol_set', default='english_basic',
                        choices=['english_basic', 'english_mandarin_basic'],
                        help='Symbols in the dataset')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Full path to the input text (phareses separated by newlines)')
    parser.add_argument('--cudnn_benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--fastpitch', type=str, default="./nvidia_fastpitch_210824.pt",
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--p_arpabet', type=float, default=1.0, help='')
    parser.add_argument('--text_cleaners', nargs='*',
                                 default=['english_cleaners_v2'], type=str,
                                 help='Type of text cleaners for input text')
    opt, unk_args = parser.parse_known_args()
    main(opt, unk_args)