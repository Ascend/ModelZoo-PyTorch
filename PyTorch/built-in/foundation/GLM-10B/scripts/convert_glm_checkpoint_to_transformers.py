# Copyright 2023 Huawei Technologies Co., Ltd
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

"""
You can use `scripts/convert_glm_checkpoint_to_transformers.py` to convert the checkpoint
```shell
python scripts/convert_glm_checkpoint_to_transformers.py CHECKPOINT_PATH MODEL_NAME
```
where `CHECKPOINT_PATH` is the path to the `mp_rank_00_model_states.pt` file,
MODEL_NAME is the repo name on huggingface hub
(should be in `["glm-large", "glm-roberta-large", "glm-large-chinese", "glm-515m", "glm-2b", "glm-10b",
"glm-10b-chinese"]`).
The `pytorch_model.bin` will be saved under the same directory as `mp_rank_00_model_states.pt`.
"""
import os
import sys
import torch


def convert_glm_checkpoint_to_transformers(checkpoint_path, copy_dict=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['module']
    if copy_dict is not None:
        word_embeddings = state_dict['word_embeddings.weight']
        for src_id, dest_id in copy_dict:
            word_embeddings[dest_id] = word_embeddings[src_id]
    directory = os.path.dirname(checkpoint_path)
    output_path = os.path.join(directory, "pytorch_model.bin")
    torch.save(state_dict, output_path)


if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    model_name = sys.argv[2]
    copy_dict = None
    assert model_name in ["glm-large", "glm-roberta-large", "glm-large-chinese", "glm-515m", "glm-2b", "glm-10b",
                          "glm-10b-chinese"]
    if model_name == "glm-10b-chinese":
        copy_dict = [(50007, 50009)]
    convert_glm_checkpoint_to_transformers(checkpoint_path, copy_dict)
