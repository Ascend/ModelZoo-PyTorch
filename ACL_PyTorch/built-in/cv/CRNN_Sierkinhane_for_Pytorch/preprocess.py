# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

import argparse
import os
import stat
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from argument_parser import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_config_argument()
    parser.add_test_image_dir_argument()
    parser.add_test_label_argument()
    return parser.parse_args()


class ImagePreprocessor:
    def __init__(self, config_, image_dir):
        self.__config = config_
        self.__image_dir = image_dir.rstrip('/')
        self.__preprocessed_image_dir = self.__get_preprocessed_image_dir()
        if not os.path.exists(self.__preprocessed_image_dir):
            os.makedirs(self.__preprocessed_image_dir)

    def __get_preprocessed_image_dir(self):
        dir_name = os.path.dirname(self.__image_dir)
        base_name = os.path.basename(self.__image_dir)
        return os.path.join(dir_name, f'preprocessed_{base_name}')

    def preprocess(self):
        image_files = os.listdir(self.__image_dir)
        for image_file in tqdm(image_files):
            self.__preprocess_single_image(os.path.join(self.__image_dir, image_file))

    def __preprocess_single_image(self, image_filepath):
        image = ImagePreprocessor.__read_image(image_filepath)
        image = self.__resize_height(image)
        image = self.__resize_width(image)
        image = self.__normalize(image)
        image = ImagePreprocessor.__reshape(image)
        ImagePreprocessor.__write_data_to_file(image, self.__get_bin_filepath(image_filepath))

    @staticmethod
    def __read_image(image_filepath):
        image = cv2.imread(image_filepath)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def __resize_height(self, image):
        height = image.shape[0]
        scale_factor = self.__config.MODEL.IMAGE_SIZE.H / height
        return cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    def __resize_width(self, image):
        scale_factor = self.__config.MODEL.IMAGE_SIZE.W / self.__config.MODEL.IMAGE_SIZE.OW
        return cv2.resize(image, (0, 0), fx=scale_factor, fy=1, interpolation=cv2.INTER_CUBIC)

    def __normalize(self, image):
        image = image.astype(np.float32)
        return (image / 255 - self.__config.DATASET.MEAN) / self.__config.DATASET.STD

    @staticmethod
    def __reshape(image):
        height, width = image.shape
        return np.reshape(image, (1, 1, height, width))

    @staticmethod
    def __write_data_to_file(image, filepath):
        image.tofile(filepath)

    def __get_bin_filepath(self, image_filepath):
        parent_dirname = os.path.dirname(self.__image_dir)
        image_dir_basename = os.path.basename(self.__image_dir)
        image_filepath_basename = os.path.basename(image_filepath)
        filename, _ = os.path.splitext(image_filepath_basename)
        return os.path.join(parent_dirname, 'preprocessed_' + image_dir_basename, filename + '.bin')


class LabelPreprocessor:
    def __init__(self, config_, label_file):
        self.__config = config_
        self.__label_file = label_file

    def preprocess(self):
        lines = LabelPreprocessor.__read_lines_from_file(self.__label_file)
        char_dict = LabelPreprocessor.__load_char_dict_from_file(self.__config.DATASET.CHAR_FILE)
        preprocessed_lines = LabelPreprocessor.__convert_lines_using_char_dict(lines, char_dict)
        LabelPreprocessor.__write_lines_to_file(
            preprocessed_lines,
            LabelPreprocessor.__get_preprocessed_filepath(self.__label_file)
        )

    @staticmethod
    def __read_lines_from_file(filepath):
        return open(filepath).read().splitlines()

    @staticmethod
    def __load_char_dict_from_file(filepath):
        char_dict = {}
        with open(filepath, 'rb') as f:
            for index, char in enumerate(f.readlines()):
                char_dict[index] = char.strip().decode('gbk', 'ignore')
        return char_dict

    @staticmethod
    def __convert_lines_using_char_dict(lines, char_dict):
        preprocessed_lines = []
        for line in lines:
            image_filepath = line.split(' ')[0]
            text = ''.join(char_dict[int(val)] for val in line.split(' ')[1:])
            preprocessed_lines.append(f'{image_filepath} {text}')
        return preprocessed_lines

    @staticmethod
    def __write_lines_to_file(lines, filepath):
        flags = os.O_WRONLY | os.O_CREAT
        mode = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(filepath, flags, mode), 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    @staticmethod
    def __get_preprocessed_filepath(filepath):
        dir_name = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        return os.path.join(dir_name, 'preprocessed_' + base_name)


if __name__ == '__main__':
    args = parse_args()
    image_preprocessor = ImagePreprocessor(config.get_config(args.config), args.image_dir)
    image_preprocessor.preprocess()
    label_preprocessor = LabelPreprocessor(config.get_config(args.config), args.label)
    label_preprocessor.preprocess()
