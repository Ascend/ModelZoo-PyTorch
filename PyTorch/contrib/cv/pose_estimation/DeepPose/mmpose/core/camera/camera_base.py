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
# ============================================================================
from abc import ABCMeta, abstractmethod

from mmcv.utils import Registry

CAMERAS = Registry('camera')


class SingleCameraBase(metaclass=ABCMeta):
    """Base class for single camera model.

    Args:
        param (dict): Camera parameters

    Methods:
        world_to_camera: Project points from world coordinates to camera
            coordinates
        camera_to_world: Project points from camera coordinates to world
            coordinates
        camera_to_pixel: Project points from camera coordinates to pixel
            coordinates
        world_to_pixel: Project points from world coordinates to pixel
            coordinates
    """

    @abstractmethod
    def __init__(self, param):
        """Load camera parameters and check validity."""

    def world_to_camera(self, X):
        """Project points from world coordinates to camera coordinates."""
        raise NotImplementedError

    def camera_to_world(self, X):
        """Project points from camera coordinates to world coordinates."""
        raise NotImplementedError

    def camera_to_pixel(self, X):
        """Project points from camera coordinates to pixel coordinates."""
        raise NotImplementedError

    def world_to_pixel(self, X):
        """Project points from world coordinates to pixel coordinates."""
        _X = self.world_to_camera(X)
        return self.camera_to_pixel(_X)
