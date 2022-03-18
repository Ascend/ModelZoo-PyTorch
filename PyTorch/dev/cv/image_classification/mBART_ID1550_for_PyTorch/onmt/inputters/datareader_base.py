# coding: utf-8
#
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
#


# several data readers need optional dependencies. There's no
# appropriate builtin exception
class MissingDependencyException(Exception):
    pass


class DataReaderBase(object):
    """Read data from file system and yield as dicts.

    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: A number
            of DataReaders need specific additional packages.
            If any are missing, this will be raised.
    """

    @classmethod
    def from_opt(cls, opt):
        """Alternative constructor.

        Args:
            opt (argparse.Namespace): The parsed arguments.
        """

        return cls()

    @classmethod
    def _read_file(cls, path):
        """Line-by-line read a file as bytes."""
        with open(path, "rb") as f:
            for line in f:
                yield line

    @staticmethod
    def _raise_missing_dep(*missing_deps):
        """Raise missing dep exception with standard error message."""
        raise MissingDependencyException(
            "Could not create reader. Be sure to install "
            "the following dependencies: " + ", ".join(missing_deps))

    def read(self, data, side, src_dir):
        """Read data from file system and yield as dicts."""
        raise NotImplementedError()
