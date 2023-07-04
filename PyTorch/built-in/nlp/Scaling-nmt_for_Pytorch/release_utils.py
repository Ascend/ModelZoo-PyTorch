# coding:utf-8
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


import argparse
from typing import Tuple


def get_next_version(release_type) -> Tuple[Tuple[int, int, int], str, str]:
    current_ver = find_version("fairseq/version.txt")
    version_list = [int(x) for x in current_ver.strip("'").split(".")]
    major, minor, patch = version_list[0], version_list[1], version_list[2]
    if release_type == "patch":
        patch += 1
    elif release_type == "minor":
        minor += 1
        patch = 0
    elif release_type == "major":
        major += 1
        minor = patch = 0
    else:
        raise ValueError(
            "Incorrect release type specified. Acceptable types are major, minor and patch."
        )

    new_version_tuple = (major, minor, patch)
    new_version_str = ".".join([str(x) for x in new_version_tuple])
    new_tag_str = "v" + new_version_str
    return new_version_tuple, new_version_str, new_tag_str


def find_version(version_file_path) -> str:
    with open(version_file_path) as f:
        version = f.read().strip()
        return version


def update_version(new_version_str) -> None:
    """
    given the current version, update the version to the
    next version depending on the type of release.
    """

    with open("fairseq/version.txt", "w") as writer:
        writer.write(new_version_str)


def main(args):
    if args.release_type in ["major", "minor", "patch"]:
        new_version_tuple, new_version, new_tag = get_next_version(args.release_type)
    else:
        raise ValueError("Incorrect release type specified")

    if args.update_version:
        update_version(new_version)

    print(new_version, new_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Versioning utils")
    parser.add_argument(
        "--release-type",
        type=str,
        required=True,
        help="type of release = major/minor/patch",
    )
    parser.add_argument(
        "--update-version",
        action="store_true",
        required=False,
        help="updates the version in fairseq/version.txt",
    )

    args = parser.parse_args()
    main(args)
