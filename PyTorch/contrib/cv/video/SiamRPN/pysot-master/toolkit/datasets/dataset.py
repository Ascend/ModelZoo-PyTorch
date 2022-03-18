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


from tqdm import tqdm


class Dataset(object):
    def __init__(self, name, dataset_root):
        self.name = name
        self.dataset_root = dataset_root
        self.videos = None

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        keys = sorted(list(self.videos.keys()))
        for key in keys:
            yield self.videos[key]

    def set_tracker(self, path, tracker_names):
        """
        Args:
            path: path to tracker results,
            tracker_names: list of tracker name
        """
        self.tracker_path = path
        self.tracker_names = tracker_names
        # for video in tqdm(self.videos.values(), 
        #         desc='loading tacker result', ncols=100):
        #     video.load_tracker(path, tracker_names)
