# Copyright 2020 Huawei Technologies Co., Ltd
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
#!/usr/bin/env python
import torch

from transformers import CamembertForMaskedLM, CamembertTokenizer


def fill_mask(masked_input, model, tokenizer, topk=5):
    # Adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py
    assert masked_input.count("<mask>") == 1
    input_ids = torch.tensor(tokenizer.encode(masked_input, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    logits = model(input_ids)[0]  # The last hidden-state is the first element of the output tuple
    masked_index = (input_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
    logits = logits[0, masked_index, :]
    prob = logits.softmax(dim=0)
    values, indices = prob.topk(k=topk, dim=0)
    topk_predicted_token_bpe = " ".join(
        [tokenizer.convert_ids_to_tokens(indices[i].item()) for i in range(len(indices))]
    )
    masked_token = tokenizer.mask_token
    topk_filled_outputs = []
    for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(" ")):
        predicted_token = predicted_token_bpe.replace("\u2581", " ")
        if " {0}".format(masked_token) in masked_input:
            topk_filled_outputs.append(
                (
                    masked_input.replace(" {0}".format(masked_token), predicted_token),
                    values[index].item(),
                    predicted_token,
                )
            )
        else:
            topk_filled_outputs.append(
                (
                    masked_input.replace(masked_token, predicted_token),
                    values[index].item(),
                    predicted_token,
                )
            )
    return topk_filled_outputs


tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
model.eval()

masked_input = "Le camembert est <mask> :)"
print(fill_mask(masked_input, model, tokenizer, topk=3))
