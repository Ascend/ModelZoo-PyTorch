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
import argparse
from nltk.translate.bleu_score import sentence_bleu


def cal_bleu_score(ground_truth_file_path, pred_file_path):
    with open(ground_truth_file_path, "rt") as g_f, open(pred_file_path, "rt") as p_f:
        sum_score = 0
        max_score = 0
        min_score = 0
        count = 0
        diff = 0
        g_line = g_f.readline().replace(",", "").replace(".", "").replace(" <unk>", "").split()
        p_line = p_f.readline().replace(",", "").replace(".", "").replace(" <unk>", "").split()
        while g_line:
            score = sentence_bleu([g_line], p_line, weights=(1, 0, 0, 0))
            if max_score < score:
                max_score = score
            if min_score > score:
                min_score = score
            sum_score += score

            count += 1

            g_line = g_f.readline().replace(",", "").replace(".", "").replace(" <unk>", "").split()
            p_line = p_f.readline().replace(",", "").replace(".", "").replace(" <unk>", "").split()

            if score != 1:
                diff += 1

    print("average bleu score is:", sum_score / count)
    print("maximum bleu score is:", max_score)
    print("minimum bleu score is:", min_score)
    print("diff num:", diff)


if __name__ == "__main__":
    """
    Usage:
    python bleu_score.py \
    --ground_truth_file_path=./pre_data/len15/test_en_len15.txt \
    --pred_file_path=./len15_online_inference_result/pred_sentence.txt
    """

    parser = argparse.ArgumentParser(description='torchtext_bleu_score.py')

    parser.add_argument('--ground_truth_file_path', required=True)
    parser.add_argument('--pred_file_path', required=True)

    opt = parser.parse_args()

    cal_bleu_score(opt.ground_truth_file_path, opt.pred_file_path)
