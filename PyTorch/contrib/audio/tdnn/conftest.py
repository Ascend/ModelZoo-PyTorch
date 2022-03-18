#     Copyright 2021 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

collect_ignore = ["setup.py"]
try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/nnet/loss/transducer_loss.py")
try:
    import fairseq  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/lobes/models/fairseq_wav2vec.py")
try:
    from transformers import Wav2Vec2Model  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/lobes/models/huggingface_wav2vec.py")
try:
    import sacrebleu  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/utils/bleu.py")
