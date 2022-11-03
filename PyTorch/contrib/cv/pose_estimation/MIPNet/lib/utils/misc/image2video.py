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

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import cv2
import os
import numpy as np


# ----------------------------------------------
IMAGE_FOLDER = '/Users/khirawal/Desktop/datasets/coco_sorted_instance/visualize'
VIDEO_NAME = os.path.join(IMAGE_FOLDER, 'video.avi')

SPF = 2.0 #seconds spent on an image
SKIP_IMAGE = 1.0 #write every nth frame to video

# ----------------------------------------------
images = sorted([img for img in os.listdir(IMAGE_FOLDER) if (img.endswith(".png") or img.endswith(".jpg"))])

frame = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(VIDEO_NAME, fourcc, fps=1/SPF, frameSize=(width, height))

for i, image in enumerate(images):
	if i%SKIP_IMAGE == 0:
		print(os.path.join(IMAGE_FOLDER, image))
		video.write(cv2.imread(os.path.join(IMAGE_FOLDER, image)))


cv2.destroyAllWindows()
video.release()

# ----------------------------------------------
