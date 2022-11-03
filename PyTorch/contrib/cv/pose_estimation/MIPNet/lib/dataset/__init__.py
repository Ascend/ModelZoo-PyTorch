# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .coco import COCODataset as coco
from .crowdpose import COCODataset as crowdpose
from .ochuman import COCODataset as ochuman


from .coco_lambda import COCODataset as coco_lambda

from .crowdpose import COCODataset as crowdpose
from .crowdpose_lambda import COCODataset as crowdpose_lambda


from .coco_lambda_012 import COCODataset as coco_lambda_012
from .coco_lambda_0123 import COCODataset as coco_lambda_0123


