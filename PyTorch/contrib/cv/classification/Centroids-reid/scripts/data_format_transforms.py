# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
def transform_bbox_s2s_to_coco(bbox):
    """Function that rearranges bbox annotations from Street2Shop format to COCO"""
    return [bbox["left"], bbox["top"], bbox["width"], bbox["height"]]


def bbox_coco_to_corners(bbox):
    # Input:
    # [top_left_x, top_left_y, width, height]
    # Output:
    # [top_left_x, top_left_y, bottom_left_x, bottom_left_y]
    bbox[0] = bbox[0]
    bbox[1] = bbox[1]
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]

    return bbox


def bbox_coco_to_center(bbox):
    # Input:
    # [top_left_x, top_left_y, width, height]
    # Output:
    # [x_center_of_bbox, y_center_of_bbox, width, height]
    bbox[0] = bbox[0] + bbox[2] / 2
    bbox[1] = bbox[1] + bbox[3] / 2
    bbox[2] = bbox[2]
    bbox[3] = bbox[3]

    return bbox


def bbox_center_to_yolo(bbox, width, height):
    # Input:
    # [x_center_of_bbox, y_center_of_bbox, width, height]
    # Output:
    # [x_center_of_bbox, y_center_of_bbox, width, height] / {image_width || image_height}
    bbox[0] = bbox[0] / width
    bbox[1] = bbox[1] / height
    bbox[2] = bbox[2] / width
    bbox[3] = bbox[3] / height

    return bbox


def bbox_yolo_to_center(bbox, width, height):
    # Input:
    # [x_center_of_bbox, y_center_of_bbox, width, height] / {image_width || image_height}
    # Output:
    # [x_center_of_bbox, y_center_of_bbox, width, height]
    bbox[0] = bbox[0] * width
    bbox[1] = bbox[1] * height
    bbox[2] = bbox[2] * width
    bbox[3] = bbox[3] * height

    return bbox


def bbox_center_to_coco(bbox):
    # Input:
    # [x_center_of_bbox, y_center_of_bbox, width, height]
    # Output:
    # [top_left_x, top_left_y, width, height]
    bbox[0] = bbox[0] - bbox[2] / 2
    bbox[1] = bbox[1] - bbox[3] / 2

    return bbox


def bbox_coco_to_yolo(bbox, width, height):
    # Input:
    # [top_left_x, top_left_y, width, height]
    # Output:
    # [x_center_of_bbox, y_center_of_bbox, width, height] / {image_width || image_height}
    bbox = bbox_coco_to_center(bbox)
    bbox = bbox_center_to_yolo(bbox, width, height)

    return bbox


def bbox_yolo_to_coco(bbox, width, height):
    # Input:
    # [x_center_of_bbox, y_center_of_bbox, width, height] / {image_width || image_height}
    # Output:
    # [top_left_x, top_left_y, width, height]
    bbox = [float(item) for item in bbox]
    bbox = bbox_yolo_to_center(bbox, width, height)
    bbox = bbox_center_to_coco(bbox)
    bbox = [int(item) for item in bbox]

    return bbox
