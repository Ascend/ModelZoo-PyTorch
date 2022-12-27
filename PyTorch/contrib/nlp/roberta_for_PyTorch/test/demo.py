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
import fairseq
import torch
import torch.nn.functional as F


def load_model():
    checkpoint_path = "./output/checkpoints"
    checkpoint_file = "checkpoint_best.pt"
    data_name_or_path = "./data/SST-2"
    model = fairseq.hub_utils.from_pretrained(
        checkpoint_path,
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_name_or_path,
        bpe="gpt2",
        load_checkpoint_heads=True,
    )["models"][0]

    return model


def test():
    loc = 'npu:0'
    torch.npu.set_device(loc)
    model = load_model()
    model = model.to(loc)
    model.eval()

    src_tokens = torch.tensor(
        [[0,   102,  2705,  2156,   157,    12, 10312, 31368,   479,     2,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
         [0,  1640,    24,   128,    29,  4839,    99, 19742,  3152,   930,
          341,   7,    28,  2156,     8,     2,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
         [0, 19650,  219,  4045,  5702,  6948,  2156, 38734, 22849,     8,
          10,  4533,   29,  5846,   127, 18137,  1217,     9,   301,    11,
          5,   885,   605,  4132,    12,  3843,  2649,  3006, 37679, 38744,
          10115, 42, 14082,  479,     2,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
         [0,   405,   128,    29,    9,     5,  1318,     9,    10, 13514,
          1368, 42792, 13,   417,  1569,   111,   411,   360,  2156,   707,
          7011,  2156, 2085, 2156,   50,    14, 31715, 19978, 21312, 21280,
          479,   2,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
         [0, 30117, 241,     8,  1808,  2650,     2,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
          1,     1,     1,     1,     1,     1,     1,     1,     1,     1]]).long().to(loc)

    output = model(src_tokens).cpu().detach()
    print(output.argmax(dim=1))


if __name__ == "__main__":
    test()
