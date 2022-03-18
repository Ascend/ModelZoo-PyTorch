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

import torch

def eval_prefetcher(load_iterator, device, pad_input=False, nhwc=False, fp16=False):
    prefetch_stream = torch.npu.Stream()

    def _prefetch():
        try:
            # Note: eval has 5 outputs, only care about 3
            img, img_id, img_size, _, _ = next(load_iterator)
            
        except StopIteration:
            return None, None, None

        with torch.npu.stream(prefetch_stream):
            img = img.to(device, non_blocking=True)
            if fp16:
                img = img.half()
            if pad_input:
                s = img.shape
                s = [s[0], 1, s[2], s[3]]
                img = torch.cat([img, torch.ones(s, device=img.device, dtype=img.dtype)], dim=1)
            if nhwc:
                img = img.permute(0, 2, 3, 1).contiguous()

        return img, img_id, img_size

    next_img, next_img_id, next_img_size = _prefetch()

    while next_img is not None:
        torch.npu.current_stream().wait_stream(prefetch_stream)
        current_img, current_img_id, current_img_size = next_img, next_img_id, next_img_size
        next_img, next_img_id, next_img_size = _prefetch()
        yield current_img, current_img_id, current_img_size

