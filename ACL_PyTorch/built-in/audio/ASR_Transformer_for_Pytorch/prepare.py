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

import os
import glob
import shutil
import logging
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file


try:
    import pandas as pd
except ImportError:
    err_msg = "The optional dependency pandas must be installed to run this recipe.\n"
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)


def prepare_aishell(data_folder, save_folder, skip_prep=False):
    """
    This function prepares the AISHELL-1 dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.

    data_folder : path to AISHELL-1 dataset.
    save_folder: path where to store the manifest csv files.
    skip_prep: If True, skip data preparation.

    """
    if skip_prep:
        return

    # If the data folders do not exist, we need to extract the data
    if not os.path.isdir(os.path.join(data_folder, "data_aishell/wav")):
        # Check for zip file and download if it doesn't exist
        zip_location = os.path.join(data_folder, "data_aishell.tgz")
        if not os.path.exists(zip_location):
            url = "https://www.openslr.org/resources/33/data_aishell.tgz"
            download_file(url, zip_location, unpack=True)
        logger.info("Extracting data_aishell.tgz...")
        shutil.unpack_archive(zip_location, data_folder)
        wav_dir = os.path.join(data_folder, "data_aishell/wav")
        tgz_list = glob.glob(wav_dir + "/*.tar.gz")
        for tgz in tgz_list:
            shutil.unpack_archive(tgz, wav_dir)
            os.remove(tgz)

    # Create filename-to-transcript dictionary
    filename2transcript = {}
    with open(
        os.path.join(
            data_folder, "data_aishell/transcript/aishell_transcript_v0.8.txt"
        ),
        "r",
    ) as f:
        lines = f.readlines()
        for line in lines:
            key = line.split()[0]
            value = " ".join(line.split()[1:])
            filename2transcript[key] = value

    splits = [
        "train",
        "dev",
        "test",
    ]
    ID_start = 0  # needed to have a unique ID for each audio
    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            continue
        logger.info("Preparing %s..." % new_filename)

        ID = []
        duration = []

        wav = []
        wav_format = []
        wav_opts = []


        transcript = []
        transcript_format = []
        transcript_opts = []

        all_wavs = glob.glob(
            os.path.join(data_folder, "data_aishell/wav") + "/" + split + "/*/*.wav"
        )
        range_n = len(all_wavs)
        for i in range(range_n):
            filename = all_wavs[i].split("/")[-1].split(".wav")[0]
            if filename not in filename2transcript:
                continue
            transcript_ = filename2transcript[filename]
            transcript.append(transcript_)
            transcript_format.append("string")
            transcript_opts.append(None)

            ID.append(ID_start + i)

            signal = read_audio(all_wavs[i])
            duration.append(signal.shape[0] / 16000)

            wav.append(all_wavs[i])
            wav_format.append("wav")
            wav_opts.append(None)



        new_df = pd.DataFrame(
            {
                "ID": ID,
                "duration": duration,
                "wav": wav,
                "transcript": transcript,
            }
        )
        new_df.to_csv(new_filename, index=False)
        ID_start += len(all_wavs)