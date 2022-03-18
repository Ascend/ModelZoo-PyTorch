# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
import librosa
import numpy as np
import soundfile

from utils import normalize

sampling_rate = 16000
top_db = 20
reduced_ratios = []

for i in range(10):
    audiopath = '../audios/audio_{}.wav'.format(i)
    print(audiopath)
    y, sr = librosa.load(audiopath)
    # Trim the beginning and ending silence
    yt, index = librosa.effects.trim(y, top_db=top_db)
    yt = normalize(yt)

    reduced_ratios.append(len(yt) / len(y))

    # Print the durations
    print(librosa.get_duration(y), librosa.get_duration(yt))
    print(len(y), len(yt))
    target = '../audios/trimed_{}.wav'.format(i)
    soundfile.write(target, yt, sampling_rate)

print('\nreduced_ratio: ' + str(100 - 100 * np.mean(reduced_ratios)))
