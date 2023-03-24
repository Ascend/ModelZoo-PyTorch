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

import argparse
import numpy as np
from espnet_onnx import Speech2Text
from tqdm import tqdm
import os,re
import time
import librosa
from pyacl.acl_infer import release_acl

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                yield fullname

def main(args):
    # define related paths
    speech2text = Speech2Text(model_dir=args.model_path, providers=["NPUExecutionProvider"], device_id=args.device_id)
    total_t = 0
    files = findAllFile(args.dataset_path)
    files = list(files)
    num = len(files)
    st = time.time()
    for fl in files:
        y, sr = librosa.load(fl, sr=16000)
        nbest = speech2text(y)
        with open(args.result_path, 'a') as f:
            res = ""
            res = "".join(nbest[0][1])
            f.write('{} {}\n'.format(fl.split('/')[-1].split('.')[0], res))
    et = time.time()
    print("wav/second:", num/(et-st))
    if speech2text.encoder_m is not None:
        speech2text.encoder_m.encoder.release_model()
    if speech2text.decoder_m is not None:
        speech2text.decoder_m.decoder.release_model()
    if speech2text.ctc_m is not None:
        speech2text.ctc_m.ctc.release_model()
    if speech2text.joint_network_m is not None:
        speech2text.joint_network_m.joint_session.release_model()
    if speech2text.lm_m is not None:
        speech2text.lm_m.lm_session.release_model()
    release_acl(args.device_id)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--dataset_path", default='test/S0768', type=str, help="datapath")
    parser.add_argument('--model_path', default="/root/.cache/espnet_onnx/asr_train_asr_qkv", type=str, help='path to the om model and config')
    parser.add_argument('--result_path', default="om.txt", type=str, help='path to result')
    parser.add_argument('--device_id', default=0, type=int, help='path to the om model and config')

    main(parser.parse_args())
