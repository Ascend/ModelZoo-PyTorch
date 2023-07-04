#!/usr/bin/env python
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
"""Helper script to compare two argparse.Namespace objects."""

from argparse import Namespace  # noqa


def main():

    ns1 = eval(input("Namespace 1: "))
    ns2 = eval(input("Namespace 2: "))

    def keys(ns):
        ks = set()
        for k in dir(ns):
            if not k.startswith("_"):
                ks.add(k)
        return ks

    k1 = keys(ns1)
    k2 = keys(ns2)

    def print_keys(ks, ns1, ns2=None):
        for k in ks:
            if ns2 is None:
                print("{}\t{}".format(k, getattr(ns1, k, None)))
            else:
                print(
                    "{}\t{}\t{}".format(k, getattr(ns1, k, None), getattr(ns2, k, None))
                )

    print("Keys unique to namespace 1:")
    print_keys(k1 - k2, ns1)
    print()

    print("Keys unique to namespace 2:")
    print_keys(k2 - k1, ns2)
    print()

    print("Overlapping keys with different values:")
    ks = [k for k in k1 & k2 if getattr(ns1, k, "None") != getattr(ns2, k, "None")]
    print_keys(ks, ns1, ns2)
    print()


if __name__ == "__main__":
    main()
