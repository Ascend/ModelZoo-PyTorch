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
from PIL import Image
import json

# sub 1 to actual value


def convert_rgb_image_to_greyscale(input_file, output_file):
    label_to_grey = {}
    label_to_rgb = {}
    rgb_to_label = {}

    with open("labels.txt", "r") as labels:
        for line in labels.readlines():
            line = line.strip()
            split_line = line.split(' ')
            if len(split_line) < 3:
                continue
            grey, label, rgb_list = split_line
            rgb = tuple(map(int, rgb_list.split(',')))

            label_to_rgb[label] = rgb
            rgb_to_label[rgb] = label
            # I forget why we are off by 1
            label_to_grey[label] = int(grey) - 1

    in_img = Image.open(input_file)
    out_img = Image.new("L", (in_img.size[0], in_img.size[1]))
    pixels = in_img.load()
    p_o = out_img.load()
    grey = (0)

    for i in range(in_img.size[0]):    # for every col:
        for j in range(in_img.size[1]):    # For every row
            # print(pixels[i, j][0:3])
            if(pixels[i, j][0:3] in rgb_to_label.keys()):
                label = rgb_to_label[pixels[i, j][0:3]]
                grey = label_to_grey[label]
            p_o[i, j] = grey
    out_img.save(output_file)


def main():
    in_file = "masterpiece.png"
    out_file = "b.png"
    convert_rgb_image_to_greyscale(in_file, out_file)


if __name__ == '__main__':
    main()
