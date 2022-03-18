# Copyright 2020 Huawei Technologies Co., Ltd
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

import sys
import json
import subprocess
from pathlib import Path

if __name__ == '__main__':
    video_dir_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    if len(sys.argv) > 3:
        dst_json_path = Path(sys.argv[3])
    else:
        dst_json_path = json_path

    with json_path.open('r') as f:
        json_data = json.load(f)

    for video_file_path in sorted(video_dir_path.iterdir()):
        file_name = video_file_path.name
        if '.mp4' not in file_name:
            continue
        name = video_file_path.stem

        ffprobe_cmd = ['ffprobe', str(video_file_path)]
        p = subprocess.Popen(
            ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()[1].decode('utf-8')

        fps = float([x for x in res.split(',') if 'fps' in x][0].rstrip('fps'))
        json_data['database'][name[2:]]['fps'] = fps

    with dst_json_path.open('w') as f:
        json.dump(json_data, f)