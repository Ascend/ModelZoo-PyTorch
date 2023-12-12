# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
import os
import torch
import numpy as np
import argparse
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset


def load_dataset(split, data_path=None, pad_length=128, combine=False):
    """Load a given dataset split (e.g., train, valid, test)."""

    assert data_path is not None, "You must specify the data path"

    # load data dictionary
    data_dict = load_dictionary(
        os.path.join(data_path, "input0", "dict.txt")
    )
    print("[input] dictionary: {} types".format(len(data_dict)))

    # load label dictionary
    label_dict = load_dictionary(
        os.path.join(data_path, "label", "dict.txt")
    )
    print("[label] dictionary: {} types".format(len(label_dict)))

    def get_path(key, split):
        return os.path.join(data_path, key, split)

    def make_dataset(key, dictionary):
        split_path = get_path(key, split)

        try:
            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                None,
                combine=combine,
            )
        except Exception as e:
            if "StorageException: [404] Path not found" in str(e):
                print(f"dataset {e} not found")
                dataset = None
            else:
                raise e
        return dataset

    input0 = make_dataset("input0", data_dict)
    assert input0 is not None, "could not find dataset: {}".format(
        get_path("input0", split)
    )
    input0 = PrependTokenDataset(input0, 0)
    src_tokens = input0

    args_seed = 1
    with data_utils.numpy_seed(args_seed):
        shuffle = np.random.permutation(len(src_tokens))

    src_tokens = maybe_shorten_dataset(
        src_tokens, split, '', 'none', 512, args_seed)

    dataset = {
        "id": IdDataset(),
        "net_input": {
            "src_tokens": RightPadDataset(
                src_tokens,
                pad_idx=data_dict.pad(),
                pad_to_length=pad_length
            ),
            "src_lengths": NumelDataset(src_tokens, reduce=False),
        },
        "nsentences": NumSamplesDataset(),
        "ntokens": NumelDataset(src_tokens, reduce=True),
    }

    if True:
        label_dataset = make_dataset("label", label_dict)
        if label_dataset is not None:
            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(
                        label_dataset,
                        id_to_strip=label_dict.eos(),
                    ),
                    offset=-label_dict.nspecial,
                )
            )
    nested_dataset = NestedDictionaryDataset(
        dataset,
        sizes=[src_tokens.sizes],
    )

    dataset = SortDataset(
        nested_dataset,
        # shuffle
        sort_order=[shuffle],
    )

    print("Loaded {0} with #samples: {1}".format(split, len(dataset)))
    return dataset


def data_tofile(dataset, batch_size, bin_path, label_path, pad_length):
    """generate data and label bin file to local"""
    
    itr = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collater,
        pin_memory=True,
        drop_last=True
    )
    labels = []
    for index, data in enumerate(itr):
        src_tokens_np = data["net_input"]["src_tokens"].numpy()
        labels.append(data["target"].numpy())
        src_tokens = os.path.join(
            bin_path, "src_tokens_" + str(index) + '.bin')
        if src_tokens_np.shape[1] > pad_length:
            src_tokens_np = src_tokens_np[:, :pad_length]
        src_tokens_np.tofile(src_tokens)
    np.array(labels).tofile(label_path)


def load_dictionary(filename):
    dictionary = Dictionary.load(filename)
    dictionary.add_symbol("<mask>")
    return dictionary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="SST-2",
                        type=str, help='dir of data')
    parser.add_argument('--data_kind', default="valid",
                        type=str, help='kind of data')
    parser.add_argument('--pad_length', default=128, type=int,
                        help='fix the pad length of one sentence')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    args = parser.parse_args()
    # generate dataset
    dataset = load_dataset(split=args.data_kind, data_path=args.data_path, pad_length=args.pad_length)
    # save data to local
    bin_path = os.path.join(args.data_path, f"roberta_base_bin_bs{args.batch_size}_pad{args.pad_length}")
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    label_path = os.path.join(args.data_path, "roberta_base.label")
    data_tofile(dataset, args.batch_size, bin_path, label_path, args.pad_length)


if __name__ == "__main__":
    main()
