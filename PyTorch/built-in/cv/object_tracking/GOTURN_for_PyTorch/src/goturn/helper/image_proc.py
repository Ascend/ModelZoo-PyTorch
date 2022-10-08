
# Copyright 2020 Huawei Technologies Co., Ltd
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


import math

import numpy as np

from .BoundingBox import BoundingBox


def cropPadImage(bbox_tight, image, dbg=False, viz=None):
    """ Around the bounding box, we define a extra context factor of 2,
    which we will crop from the original image
    """

    pad_image_location = computeCropPadImageLocation(bbox_tight, image)
    roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
    roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))
    roi_width = min(image.shape[1], max(1.0, math.ceil(pad_image_location.x2 - pad_image_location.x1)))
    roi_height = min(image.shape[0], max(1.0, math.ceil(pad_image_location.y2 - pad_image_location.y1)))

    err = 0.000000001  # To take care of floating point arithmetic errors
    cropped_image = image[int(roi_bottom + err):int(roi_bottom + roi_height), int(roi_left + err):int(roi_left + roi_width)]
    output_width = max(math.ceil(bbox_tight.compute_output_width()), roi_width)
    output_height = max(math.ceil(bbox_tight.compute_output_height()), roi_height)
    if image.ndim > 2:
        output_image = np.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
    else:
        output_image = np.zeros((int(output_height), int(output_width)), dtype=image.dtype)

    edge_spacing_x = min(bbox_tight.edge_spacing_x(), (image.shape[1] - 1))
    edge_spacing_y = min(bbox_tight.edge_spacing_y(), (image.shape[0] - 1))

    # rounding should be done to match the width and height
    output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0], int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image

    return output_image, pad_image_location, edge_spacing_x, edge_spacing_y


def computeCropPadImageLocation(bbox_tight, image):
    """Get the valid image coordinates for the context region in target
    or search region in full image
    """

    # Center of the bounding box
    bbox_center_x = bbox_tight.get_center_x()
    bbox_center_y = bbox_tight.get_center_y()

    image_height = image.shape[0]
    image_width = image.shape[1]

    # Padded output width and height
    output_width = bbox_tight.compute_output_width()
    output_height = bbox_tight.compute_output_height()

    roi_left = max(0.0, bbox_center_x - (output_width / 2.))
    roi_bottom = max(0.0, bbox_center_y - (output_height / 2.))

    # New ROI width
    # -------------
    # 1. left_half should not go out of bound on the left side of the
    # image
    # 2. right_half should not go out of bound on the right side of the
    # image
    left_half = min(output_width / 2., bbox_center_x)
    right_half = min(output_width / 2., image_width - bbox_center_x)
    roi_width = max(1.0, left_half + right_half)

    # New ROI height
    # Similar logic applied that is applied for 'New ROI width'
    top_half = min(output_height / 2., bbox_center_y)
    bottom_half = min(output_height / 2., image_height - bbox_center_y)
    roi_height = max(1.0, top_half + bottom_half)

    # Padded image location in the original image
    objPadImageLocation = BoundingBox(roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height)

    return objPadImageLocation
