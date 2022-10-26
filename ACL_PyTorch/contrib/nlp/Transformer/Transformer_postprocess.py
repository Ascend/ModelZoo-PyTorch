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
import glob
import numpy as np
import dill as pickle
import argparse
import transformer.Constants as Constants


def postprocess(bin_file_path, data_pkl, result_path):
    data = pickle.load(open(data_pkl, 'rb'))
    TRG = data['vocab']['trg']

    bin_file_list = glob.glob(os.path.join(bin_file_path, "*.bin"))
    bin_file_num = len(bin_file_list)

    if os.path.exists(result_path) == False:
        os.mkdir(result_path)

    f_pred = open(os.path.join(result_path, "pred_sentence.txt"), "wt")
    f_pred_array = open(os.path.join(result_path, "pred_sentence_array.txt"), "wt")
    for i in range(bin_file_num):
        pred_seq = np.fromfile(os.path.join(bin_file_path, str(i) + "_0.bin"), dtype=np.int64)
        pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
        pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
        f_pred.write(pred_line.lstrip() + "\n")
        f_pred_array.write(str(pred_seq) + "\n")

    f_pred.close()
    f_pred_array.close()


if __name__ == "__main__":
    """
    Usage Example:
    python postprocess.py \
    --bin_file_path ./result/dumpOutput_device0 \
    --data_pkl ./pkl_file/m30k_deen_shr.pkl \
    --result_path len15_benchmark_inference_result
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--bin_file_path', required=True,
                        help='benchmark inference result path')
    parser.add_argument('--data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('--result_path', required=True, help='result path')

    opt = parser.parse_args()

    print("Processing...")

    postprocess(opt.bin_file_path, opt.data_pkl, opt.result_path)

    print("Done!")
