# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

from GooglePretrainedWeightDownloader import GooglePretrainedWeightDownloader
from NVIDIAPretrainedWeightDownloader import NVIDIAPretrainedWeightDownloader
from WikiDownloader import WikiDownloader
from BooksDownloader import BooksDownloader
from GLUEDownloader import GLUEDownloader
from SquadDownloader import SquadDownloader


class Downloader:

    def __init__(self, dataset_name, save_path):
        self.dataset_name = dataset_name
        self.save_path = save_path

    def download(self):
        if self.dataset_name == 'bookscorpus':
            self.download_bookscorpus()

        elif self.dataset_name == 'wikicorpus_en':
            self.download_wikicorpus('en')

        elif self.dataset_name == 'wikicorpus_zh':
            self.download_wikicorpus('zh')

        elif self.dataset_name == 'google_pretrained_weights':
            self.download_google_pretrained_weights()

        elif self.dataset_name == 'nvidia_pretrained_weights':
            self.download_nvidia_pretrained_weights()

        elif self.dataset_name in {'mrpc', 'sst-2'}:
            self.download_glue(self.dataset_name)

        elif self.dataset_name == 'squad':
            self.download_squad()

        elif self.dataset_name == 'all':
            self.download_bookscorpus()
            self.download_wikicorpus('en')
            self.download_wikicorpus('zh')
            self.download_google_pretrained_weights()
            self.download_nvidia_pretrained_weights()
            self.download_glue('mrpc')
            self.download_glue('sst-2')
            self.download_squad()

        else:
            print(self.dataset_name)
            assert False, 'Unknown dataset_name provided to downloader'

    def download_bookscorpus(self):
        downloader = BooksDownloader(self.save_path)
        downloader.download()

    def download_wikicorpus(self, language):
        downloader = WikiDownloader(language, self.save_path)
        downloader.download()

    def download_google_pretrained_weights(self):
        downloader = GooglePretrainedWeightDownloader(self.save_path)
        downloader.download()

    def download_nvidia_pretrained_weights(self):
        downloader = NVIDIAPretrainedWeightDownloader(self.save_path)
        downloader.download()

    def download_glue(self, task_name):
        downloader = GLUEDownloader(self.save_path)
        downloader.download(task_name)

    def download_squad(self):
        downloader = SquadDownloader(self.save_path)
        downloader.download()
