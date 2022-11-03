# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from core.inference import get_max_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.function import AverageMeter


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
def cutout(input_foreground, segmentation_foreground, input_context, segmentation_context, paste_loc_x=None, paste_loc_y=None):
    C, H, W = input_foreground.size()

    if paste_loc_x is None:
        paste_loc_x = np.random.randint(2*W)
        paste_loc_y = np.random.randint(2*H)

    segmentation_background = 1.0*(segmentation_foreground[:, :]==0)

    # ------------------------------
    segmentation_foreground_idx = (segmentation_foreground == 1).nonzero() 
    segmentation_background_idx = (segmentation_background == 1).nonzero() 
    segmentation_context_idx = (segmentation_context == 1).nonzero() 

    # ------------------------------
    canvas = torch.zeros((C, 3*H, 3*W))

    ## we have 3 layers, first the background of A, then foreground of B, then foreground A
    ## copy input foreground pixels

    # ## paste background of A
    origin_x = W; origin_y = H 
    segmentation_idx = segmentation_background_idx
    canvas[:, origin_y+segmentation_idx[:, 0], origin_x+segmentation_idx[:, 1]] = input_foreground[:, segmentation_idx[:, 0], segmentation_idx[:, 1]]

    # ## paste foreground of B, is the top left of B in the canvas
    origin_x = paste_loc_x; origin_y = paste_loc_y 
    segmentation_idx = segmentation_context_idx
    canvas[:, origin_y+segmentation_idx[:, 0], origin_x+segmentation_idx[:, 1]] = input_context[:, segmentation_idx[:, 0], segmentation_idx[:, 1]]

    ## paste foreground of A
    origin_x = W; origin_y = H 
    segmentation_idx = segmentation_foreground_idx
    canvas[:, origin_y+segmentation_idx[:, 0], origin_x+segmentation_idx[:, 1]] = input_foreground[:, segmentation_idx[:, 0], segmentation_idx[:, 1]]

    # ------------------------------
    origin_x = W; origin_y = H 
    input1 = canvas[:, origin_y:origin_y+H, origin_x:origin_x+W]

    # ------------------------------
    origin_x = paste_loc_x; origin_y = paste_loc_y 
    input2 = canvas[:, origin_y:origin_y+H, origin_x:origin_x+W]
    
    return input1, input2, canvas

# ---------------------------------------------------------------------------------


def cutout_data(input, segmentation, paste_loc_x=None, paste_loc_y=None):
    return cutout(input_foreground=input[0], segmentation_foreground=segmentation[0], input_context=input[1], segmentation_context=segmentation[1], paste_loc_x=paste_loc_x, paste_loc_y=paste_loc_y)















































