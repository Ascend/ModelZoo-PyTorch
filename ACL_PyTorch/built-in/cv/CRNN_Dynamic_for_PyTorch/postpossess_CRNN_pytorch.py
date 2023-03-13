# Copyright 2022 Huawei Technologies Co., Ltd
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

import time
import argparse
import numpy as np
import torch
from ais_bench.infer.interface import InferSession

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        """

        length = []
        result = []
        for item in text:
            length.append(len(item))
            r = []
            for char in item:
                index = self.dict[char]
                r.append(index)
            result.append(r)
        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)
        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)
        text = result_temp
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "does not match"
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "does not match declared length"
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts


total_img = 3000

def get_Acc(input_npy, label, session):
    # label
    keys, vals = [], []
    with open(label, 'r') as f:
        content = f.read()
        contents = content.split('\n')[:-1]
    for cot in contents:
        cot = cot.split(':')
        keys.append(cot[0])
        vals.append(cot[1])

    labels = dict(zip(keys, vals))
    count = 0
    t = 0
    for index in range(total_img):
        index += 1

        data = np.load(f'{input_npy}/test_{index}.npy')
        t0 = time.time()
        preds = session.infer([data], "dymshape", custom_sizes=10000000)[0]
        t += (time.time() - t0)
        preds = torch.from_numpy(preds)
        preds_size = torch.LongTensor([preds.size(0)])

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        converter = strLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        key = 'test_{}.npy'.format(index)
        if sim_preds == labels[key]:
            count += 1
        else:
            print("label:{}  pred:{}".format(labels[key], sim_preds))

    # acc
    print('*'*50)
    print('rightNum: {}'.format(count))
    print('totalNum: {}'.format(total_img))
    print("accuracy_rate %.2f" % (count / total_img * 100))
    print(f'E2E time(s): {t}s')
    print('*'*50)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npy', default="./input_npy", type=str,
                        help="input path")
    parser.add_argument('--label', default="./label.txt",
                        type=str, help='label path')
    parser.add_argument('--device', default=0,
                        type=int, help='npu device id')
    parser.add_argument('--om_path', default="./om/crnn_dy.om",
                        type=str, help='om path')                        
   
    args = parser.parse_args()
    global_session = InferSession(args.device, args.om_path)

    get_Acc(args.input_npy, args.label, global_session)
