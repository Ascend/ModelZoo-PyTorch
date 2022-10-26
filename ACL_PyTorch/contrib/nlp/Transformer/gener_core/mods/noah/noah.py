# -*- coding:utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import numpy as np
import os

from ...mod_modify.interface import BaseGraph
from ...mod_modify.interface import AttrType as AT
from ...mod_uti import mod_uti as mu
from ...mod_uti import mod_param_uti
from ...mod_uti import om_uti
from ...mod_uti.log_uti import mlog


class Keys(object):
    batch = "batch"
    in_img_name = "input_img_name"
    task_0_decode_0 = "task_0_decode_0"
    task_0_decode_1 = "task_0_decode_1"
    task_0_decode_2 = "task_0_decode_2"
    task_1_decode_0 = "task_1_decode_0"
    task_1_decode_1 = "task_1_decode_1"
    task_0_clip_boxes = "task_0_clip_boxes"
    task_1_clip_boxes = "task_1_clip_boxes"
    task_0_nms = "task_0_nms"
    task_1_nms = "task_1_nms"
    in_img_h = "input_img_h"
    in_img_w = "input_img_w"
    filter_thresh = "filter_thresh"
    score_thresh = "score_thresh"
    max_output_size = "max_output_size"
    rpn_iou_thresh = "rpn_iou_thresh"
    fast_iou_thresh = "fast_iou_thresh"
    task0_merge = "task0_merge"
    task1_merge = "task1_merge"
    task0_rpn_nms = "task_0_rpn"
    task1_rpn_nms = "task_1_rpn"
    task0_rpn_merge = "task0_rpn_merge"
    task1_rpn_merge = "task1_rpn_merge"
    task0_rpn_slice = "task0_rpn_slice"
    task1_rpn_slice = "task1_rpn_slice"
    rpn_slice_begin = "rpn_slice_begin"
    rpn_slice_end = "rpn_slice_end"
    task0_rpn_clip_boxes = "task0_rpn_clip_boxes"
    task1_rpn_clip_boxes = "task1_rpn_clip_boxes"
    del_shape_node = "del_shape_node"
    task_0_roi_0 = "task_0_roi_0"
    task_0_roi_1 = "task_0_roi_1"
    task_1_roi_0 = "task_1_roi_0"
    pool_height = "pool_height"
    pool_width = "pool_width"


class Support(object):
    # 3rdmod releated
    framework = (mod_param_uti.ModFMK.TF,)
    # om releated
    om_gen_bin = {"om_gen_bin": ("atc",)}
    atc_params = {"soc_version": ("Ascend610",)}  # Ascend710 to verify


def check_mod_params(params: dict):
    check_items = {Keys.batch: {"dtype": int, "min": 0, "max": 2},
                   Keys.in_img_name: {"dtype": str, "min": 0},
                   Keys.task_0_decode_0: {"dtype": str, "min": 0},
                   Keys.task_0_decode_1: {"dtype": str, "min": 0},
                   Keys.task_0_decode_2: {"dtype": str, "min": 0},
                   Keys.task_1_decode_0: {"dtype": str, "min": 0},
                   Keys.task_1_decode_1: {"dtype": str, "min": 0},
                   Keys.task_0_clip_boxes: {"dtype": str, "min": 0},
                   Keys.task_1_clip_boxes: {"dtype": str, "min": 0},
                   Keys.task0_rpn_nms: {"dtype": str, "min": 0},
                   Keys.task1_rpn_nms: {"dtype": str, "min": 0},
                   Keys.task_0_nms: {"dtype": str, "min": 0},
                   Keys.task_1_nms: {"dtype": str, "min": 0},
                   Keys.del_shape_node: {"dtype": str, "min": 0},
                   Keys.task_0_roi_0: {"dtype": str, "min": 0},
                   Keys.task_0_roi_1: {"dtype": str, "min": 0},
                   Keys.task_1_roi_0: {"dtype": str, "min": 0},
                   Keys.in_img_h: {"dtype": int, "min": 0},
                   Keys.in_img_w: {"dtype": int, "min": 0},
                   Keys.pool_height: {"dtype": int, "min": 0},
                   Keys.pool_width: {"dtype": int, "min": 0},
                   Keys.filter_thresh: {"dtype": float, "min": 0.0},
                   Keys.score_thresh: {"dtype": float, "min": 0.0,
                                       "max": 1.0}}

    if (not mod_param_uti.check_params_dtype_len(params, check_items)):
        return False

    return True


def set_inputs(mod: BaseGraph, params: dict):
    input_img_node = mod.get_node(params.get(Keys.in_img_name))
    in_img_h = params.get(Keys.in_img_h)
    in_img_w = params.get(Keys.in_img_w)
    input_img_node.set_shape([in_img_h, in_img_w, 3])
    mlog("set {} shape:({}, {}, 3)".format(Keys.in_img_name,
                                           in_img_h, in_img_w),
         level=logging.INFO)

    return True


def add_decode(mod: BaseGraph, node_name):
    cmp_value = 192.0  # the max number of exponent arithmetic
    return mod.add_new_node(node_name, "Decode",
                            {"scale_factor1": (AT.FLOAT, 1.0),
                             "scale_factor2": (AT.FLOAT, 1.0),
                             "scale_factor3": (AT.FLOAT, 1.0),
                             "scale_factor4": (AT.FLOAT, 1.0),
                             "cmp_value": (AT.FLOAT, cmp_value)})


def get_scope_func(node, scope):
    return node.name.startswith(scope + '/')


def add_decode_trans(mod: BaseGraph, node_name):
    decode_trans = mod.add_new_node(node_name, "Transpose",
                                    {"Tperm": (AT.DTYPE, "int32"),
                                     "T": (AT.DTYPE, "float32")})
    return decode_trans


def add_decode_boxes_trans(mod):
    task0_fast_boxes_trans = mod.add_new_node("task0_fast_boxes_trans",
                                              "Transpose",
                                              {"Tperm": (AT.DTYPE, "int32"),
                                               "T": (AT.DTYPE, "float32")})
    task0_fast_anchors_trans = mod.add_new_node("task0_fast_anchors_trans",
                                                "Transpose",
                                                {"Tperm": (AT.DTYPE, "int32"),
                                                 "T": (AT.DTYPE, "float32")})
    task1_fast_boxes_trans = mod.add_new_node("task1_fast_boxes_trans",
                                              "Transpose",
                                              {"Tperm": (AT.DTYPE, "int32"),
                                               "T": (AT.DTYPE, "float32")})
    task1_fast_anchors_trans = mod.add_new_node("task1_fast_anchors_trans",
                                                "Transpose",
                                                {"Tperm": (AT.DTYPE, "int32"),
                                                 "T": (AT.DTYPE, "float32")})

    return (task0_fast_boxes_trans, task0_fast_anchors_trans,
            task1_fast_boxes_trans, task1_fast_anchors_trans)


def add_decode_reshape(mod):
    task0_fast_boxes_reshape = mod.add_new_node("task0_fast_boxes_reshape",
                                                "Reshape",
                                                {"Tshape": (AT.DTYPE, "int32"),
                                                 "T": (AT.DTYPE, "float32")})
    task0_fast_anchors_reshape = mod.add_new_node("task0_fast_anchors_reshape",
                                                  "Reshape",
                                                  {"Tshape":
                                                   (AT.DTYPE, "int32"),
                                                   "T": (AT.DTYPE, "float32")})
    task1_fast_boxes_reshape = mod.add_new_node("task1_fast_boxes_reshape",
                                                "Reshape",
                                                {"Tshape": (AT.DTYPE, "int32"),
                                                 "T": (AT.DTYPE, "float32")})
    task1_fast_anchors_reshape = mod.add_new_node("task1_fast_anchors_reshape",
                                                  "Reshape",
                                                  {"Tshape":
                                                   (AT.DTYPE, "int32"),
                                                   "T": (AT.DTYPE, "float32")})

    return (task0_fast_boxes_reshape, task0_fast_anchors_reshape,
            task1_fast_boxes_reshape, task1_fast_anchors_reshape)


def replace_task0_norm_decode(mod, params, decode_trans_idx):
    task0_decode0 = add_decode(mod, Keys.task_0_decode_0)
    mod.replace_scope(params.get(Keys.task_0_decode_0), task0_decode0,
                      get_scope_func, ["task_0/rpn_big/Reshape",
                                       "anchors/fm_anchors"])
    task0_decode0_boxes_trans = add_decode_trans(mod,
                                                 "task0_decode0_boxes_trans")
    mod.node_add_behind("task_0/rpn_big/Reshape", task0_decode0_boxes_trans)
    task0_decode0_boxes_trans.set_input_node(1, [decode_trans_idx.name])
    task0_decode0_anchors_trans = add_decode_trans(
        mod, "task0_decode0_anchors_trans")
    mod.node_add_behind("anchors/fm_anchors", task0_decode0_anchors_trans)
    task0_decode0_anchors_trans.set_input_node(1, [decode_trans_idx.name])

    task0_decode1 = add_decode(mod, Keys.task_0_decode_1)
    mod.replace_scope(params.get(Keys.task_0_decode_1), task0_decode1,
                      get_scope_func, ["task_0/rpn_small/Reshape",
                                       "anchors/fm_anchors_1"])
    task0_decode1_boxes_trans = add_decode_trans(mod,
                                                 "task0_decode1_boxes_trans")
    mod.node_add_behind("task_0/rpn_small/Reshape", task0_decode1_boxes_trans)
    task0_decode1_boxes_trans.set_input_node(1, [decode_trans_idx.name])
    task0_decode1_anchors_trans = add_decode_trans(
        mod, "task0_decode1_anchors_trans")
    mod.node_add_behind("anchors/fm_anchors_1", task0_decode1_anchors_trans)
    task0_decode1_anchors_trans.set_input_node(1, [decode_trans_idx.name])


def replace_task1_norm_decode(mod, params, decode_trans_idx):
    task1_decode0 = add_decode(mod, Keys.task_1_decode_0)
    mod.replace_scope(params.get(Keys.task_1_decode_0), task1_decode0,
                      get_scope_func, ["task_1/rpn_big/Reshape",
                                       "anchors/fm_anchors_2"])
    task1_decode0_boxes_trans = add_decode_trans(mod,
                                                 "task1_decode0_boxes_trans")
    mod.node_add_behind("task_1/rpn_big/Reshape", task1_decode0_boxes_trans)
    task1_decode0_boxes_trans.set_input_node(1, [decode_trans_idx.name])
    task1_decode0_anchors_trans = add_decode_trans(
        mod, "task1_decode0_anchors_trans")
    mod.node_add_behind("anchors/fm_anchors_2", task1_decode0_anchors_trans)
    task1_decode0_anchors_trans.set_input_node(1, [decode_trans_idx.name])


def replace_decode(mod: BaseGraph, params: dict):
    decode_trans_idx = mod.add_const_node("decode_trans_idx",
                                          np.array([0, 3, 1, 2]).astype(
                                              np.int32))
    replace_task0_norm_decode(mod, params, decode_trans_idx)

    task0_decode2 = add_decode(mod, Keys.task_0_decode_2)
    mod.replace_scope(params.get(Keys.task_0_decode_2), task0_decode2,
                      get_scope_func, ["task_0/truediv", "task_0/Tile"])
    fast_trans_idx = mod.add_const_node("fast_trans_idx",
                                        np.array([2, 1, 0]).astype(np.int32))
    fast_max_output = mod.get_node("task_0/fastrcnn_predictions/map/while/"
                                   "non_max_suppression/"
                                   "NonMaxSuppressionV2/max_output_size")
    fast_max_output_size = fast_max_output.const_value
    fast_boxes_shape = mod.add_const_node("fast_boxes_shape",
                                          np.array([params.get(Keys.batch),
                                                    4, -1,
                                                    fast_max_output_size]).
                                          astype(np.int32))
    (task0_fast_boxes_trans, task0_fast_anchors_trans,
     task1_fast_boxes_trans, task1_fast_anchors_trans) = add_decode_boxes_trans(mod)
    (task0_fast_boxes_reshape, task0_fast_anchors_reshape,
     task1_fast_boxes_reshape, task1_fast_anchors_reshape) = add_decode_reshape(mod)
    mod.node_add_behind("task_0/truediv", task0_fast_boxes_trans.name)
    task0_fast_boxes_trans.set_input_node(1, [fast_trans_idx.name])
    mod.node_add_behind("task_0/Tile", task0_fast_anchors_trans.name)
    task0_fast_anchors_trans.set_input_node(1, [fast_trans_idx.name])
    mod.node_add_behind(task0_fast_boxes_trans.name,
                        task0_fast_boxes_reshape.name)
    task0_fast_boxes_reshape.set_input_node(1, [fast_boxes_shape.name])
    mod.node_add_behind(task0_fast_anchors_trans.name,
                        task0_fast_anchors_reshape.name)
    task0_fast_anchors_reshape.set_input_node(1, [fast_boxes_shape.name])

    replace_task1_norm_decode(mod, params, decode_trans_idx)

    task1_decode1 = add_decode(mod, Keys.task_1_decode_1)
    mod.replace_scope(params.get(Keys.task_1_decode_1), task1_decode1,
                      get_scope_func, ["task_1/truediv", "task_1/Tile"])

    mod.node_add_behind("task_1/truediv", task1_fast_boxes_trans.name)
    task1_fast_boxes_trans.set_input_node(1, [fast_trans_idx.name])
    mod.node_add_behind("task_1/Tile", task1_fast_anchors_trans.name)
    task1_fast_anchors_trans.set_input_node(1, [fast_trans_idx.name])
    mod.node_add_behind(task1_fast_boxes_trans.name,
                        task1_fast_boxes_reshape.name)
    task1_fast_boxes_reshape.set_input_node(1, [fast_boxes_shape.name])
    mod.node_add_behind(task1_fast_anchors_trans.name,
                        task1_fast_anchors_reshape.name)
    task1_fast_anchors_reshape.set_input_node(1, [fast_boxes_shape.name])

    return True


def add_clip_boxes(mod: BaseGraph, node_name, params):
    return mod.add_new_node(node_name, "ClipToWindow", {
        "clip_y_min": (AT.FLOAT, 0),
        "clip_x_min": (AT.FLOAT, 0),
        "clip_y_max": (AT.FLOAT, params.get(Keys.in_img_w)),
        "clip_x_max": (AT.FLOAT, params.get(Keys.in_img_h))})


def replace_clip_boxes(mod: BaseGraph, params: dict):
    task0_clip_boxes = add_clip_boxes(mod, Keys.task_0_clip_boxes, params)
    mod.replace_scope(params.get(Keys.task_0_clip_boxes), task0_clip_boxes,
                      get_scope_func, [Keys.task_0_decode_2])
    task1_clip_boxes = add_clip_boxes(mod, Keys.task_1_clip_boxes, params)
    mod.replace_scope(params.get(Keys.task_1_clip_boxes), task1_clip_boxes,
                      get_scope_func, [Keys.task_1_decode_1])
    return True


def add_nms(mod: BaseGraph, node_name, params, iou_threshold, score_threshold,
            max_output_size):
    max_sort_num = 3968  # the max sort num of topK on davinci
    return mod.add_new_node(node_name,
                            "BatchMultiClassNms",
                            {"max_output_size": (AT.INT, max_output_size),
                             "iou_threshold": (AT.FLOAT, iou_threshold),
                             "score_threshold": (AT.FLOAT, score_threshold),
                             "weights": (AT.LIST_FLOAT,
                                         [1.0 / params.get(Keys.in_img_w),
                                          1.0 / params.get(Keys.in_img_h),
                                          1.0 / params.get(Keys.in_img_w),
                                          1.0 / params.get(Keys.in_img_h)]),
                             "biases":
                             (AT.LIST_FLOAT, [-16.0, -16.0, -16.0, -16.0]),
                             "sort_num":
                             (AT.INT, max_sort_num)})


def add_merge(mod: BaseGraph, node_name):
    return mod.add_new_node(node_name,
                            "MergeProposal", {})


def adjust_type(mod):
    task1_mul_y = mod.get_node("task_1/mul/y")
    task1_mul_y.set_const_value(np.array([0.03125]).astype(np.float16))
    task0_mul_y = mod.get_node("task_0/mul/y")
    task0_mul_y.set_const_value(np.array([0.03125]).astype(np.float16))
    task0_mul_1_y = mod.get_node("task_0/mul_1/y")
    task0_mul_1_y.set_const_value(np.array([0.125]).astype(np.float16))


def trans_decode_res(mod):
    decode_res_trans_idx = mod.add_const_node("decode_res_trans_idx",
                                              np.array([1, 0, 2, 3]).astype(
                                                  np.int32))
    task0_decode0_res_trans = mod.add_new_node("task0_decode0_res_trans",
                                               "Transpose",
                                               {"Tperm": (AT.DTYPE, "int32"),
                                                "T": (AT.DTYPE, "float32")})
    mod.node_add_behind("task_0_decode_0", task0_decode0_res_trans.name)
    task0_decode0_res_trans.set_input_node(1, [decode_res_trans_idx.name])

    task0_decode1_res_trans = mod.add_new_node("task0_decode1_res_trans",
                                               "Transpose",
                                               {"Tperm": (AT.DTYPE, "int32"),
                                                "T": (AT.DTYPE, "float32")})
    mod.node_add_behind("task_0_decode_1", task0_decode1_res_trans.name)
    task0_decode1_res_trans.set_input_node(1, [decode_res_trans_idx.name])

    task1_decode0_res_trans = mod.add_new_node("task1_decode0_res_trans",
                                               "Transpose",
                                               {"Tperm": (AT.DTYPE, "int32"),
                                                "T": (AT.DTYPE, "float32")})
    mod.node_add_behind("task_1_decode_0", task1_decode0_res_trans.name)
    task1_decode0_res_trans.set_input_node(1, [decode_res_trans_idx.name])


def reshape_decode_res(mod, params):
    task0_reshape_shape = mod.get_node("task_0/Reshape/shape")
    task0_reshape_shape.set_const_value(np.array([4, -1]).astype(np.int32))
    task0_reshape_2_shape = mod.get_node("task_0/Reshape_2/shape")
    task0_reshape_2_shape.set_const_value(np.array([4, -1]).astype(np.int32))

    task0_concat_axis = mod.get_node("task_0/concat/axis")
    task0_concat_axis.set_const_value(np.array([1]).astype(np.int32))

    task1_reshape_shape = mod.get_node("task_1/Reshape/shape")
    task1_reshape_shape.set_const_value(np.array([4, -1]).astype(np.int32))

    rpn_boxes_shape = mod.add_const_node("rpn_boxes_shape",
                                         np.array(
                                             [params.get(Keys.batch), 4, 1,
                                              -1]).astype(np.int32))
    rpn_scores_shape = mod.add_const_node("rpn_scores_shape",
                                          np.array(
                                              [params.get(Keys.batch), 1, 1,
                                               -1]).astype(np.int32))
    task0_rpn_reshape_boxes = mod.add_new_node("task0_rpn_reshape_boxes",
                                               "Reshape",
                                               {"T": (AT.DTYPE, "float32"),
                                                "Tshape": (AT.DTYPE, "int32")})
    mod.node_add_behind("task_0/concat", task0_rpn_reshape_boxes.name)
    task0_rpn_reshape_boxes.set_input_node(1, [rpn_boxes_shape.name])

    task0_rpn_reshape_scores = mod.add_new_node("task0_rpn_reshape_scores",
                                                "Reshape",
                                                {"T": (AT.DTYPE, "float32"),
                                                 "Tshape": (
                                                     AT.DTYPE, "int32")})
    mod.node_add_behind("task_0/concat_1", task0_rpn_reshape_scores.name)
    task0_rpn_reshape_scores.set_input_node(1, [rpn_scores_shape.name])

    task1_rpn_reshape_boxes = mod.add_new_node("task1_rpn_reshape_boxes",
                                               "Reshape",
                                               {"T": (AT.DTYPE, "float32"),
                                                "Tshape": (AT.DTYPE, "int32")})
    mod.node_add_behind("task_1/Reshape", task1_rpn_reshape_boxes.name)
    task1_rpn_reshape_boxes.set_input_node(1, [rpn_boxes_shape.name])

    task1_rpn_reshape_scores = mod.add_new_node("task1_rpn_reshape_scores",
                                                "Reshape",
                                                {"T": (AT.DTYPE, "float32"),
                                                 "Tshape": (
                                                     AT.DTYPE, "int32")})
    mod.node_add_behind("task_1/Reshape_1", task1_rpn_reshape_scores.name)
    task1_rpn_reshape_scores.set_input_node(1, [rpn_scores_shape.name])
    task0_rpn_clip_boxes = add_clip_boxes(mod, Keys.task0_rpn_clip_boxes,
                                          params)
    mod.node_add_behind(task0_rpn_reshape_boxes.name,
                        task0_rpn_clip_boxes.name)
    task1_rpn_clip_boxes = add_clip_boxes(mod, Keys.task1_rpn_clip_boxes,
                                          params)
    mod.node_add_behind(task1_rpn_reshape_boxes.name,
                        task1_rpn_clip_boxes.name)


def replace_task0_rpn(mod, params, rpn_slice_begin, rpn_slice_end,
                      rpn_trans_idx, rpn_reshape_shape):
    task0_rpn_max_output = mod.get_node("task_0/generate_rpn_proposals/"
                                        "non_max_suppression/"
                                        "NonMaxSuppressionV2/"
                                        "max_output_size")
    task0_rpn_max_output_size = task0_rpn_max_output.const_value
    task0_rpn_iou = mod.get_node("task_0/generate_rpn_proposals/"
                                 "non_max_suppression/iou_threshold")
    task0_rpn_iou_thresh = task0_rpn_iou.const_value
    task0_rpn_nms = add_nms(mod, Keys.task0_rpn_nms, params,
                            task0_rpn_iou_thresh,
                            params.get(Keys.filter_thresh),
                            task0_rpn_max_output_size)
    mod.replace_scope(params.get(Keys.task0_rpn_nms), task0_rpn_nms,
                      get_scope_func, ["task_0/concat", "task_0/concat_1"])
    task0_rpn_merge = add_merge(mod, Keys.task0_rpn_merge)
    mod.node_add_behind(task0_rpn_nms, task0_rpn_merge)
    task0_rpn_slice = mod.add_new_node(Keys.task0_rpn_slice, "Slice",
                                       {"T": (AT.DTYPE, "float32"),
                                        "Index": (AT.DTYPE, "int32")})
    mod.node_add_behind(task0_rpn_merge, task0_rpn_slice)
    task0_rpn_slice.set_input_node(1,
                                   [rpn_slice_begin.name, rpn_slice_end.name])

    task0_rpn_trans = mod.add_new_node("task0_rpn_trans", "Transpose",
                                       {"Tperm": (AT.DTYPE, "int32"),
                                        "T": (AT.DTYPE, "float32")})
    task0_rpn_reshape = mod.add_new_node("task0_rpn_reshape", "Reshape",
                                         {"T": (AT.DTYPE, "float32"),
                                          "Tshape": (AT.DTYPE, "int32")})
    mod.node_add_behind(task0_rpn_slice, task0_rpn_trans)
    mod.node_add_behind(task0_rpn_trans, task0_rpn_reshape)
    task0_rpn_trans.set_input_node(1, [rpn_trans_idx.name])
    task0_rpn_reshape.set_input_node(1, [rpn_reshape_shape.name])


def replace_task1_rpn(mod, params, rpn_slice_begin, rpn_slice_end,
                      rpn_trans_idx, rpn_reshape_shape):
    task1_rpn_max_output = mod.get_node("task_1/generate_rpn_proposals/"
                                        "non_max_suppression/"
                                        "NonMaxSuppressionV2/max_output_size")
    task1_rpn_max_output_size = task1_rpn_max_output.const_value
    task1_rpn_iou = mod.get_node("task_1/generate_rpn_proposals/"
                                 "non_max_suppression/iou_threshold")
    task1_rpn_iou_thresh = task1_rpn_iou.const_value
    task1_rpn_nms = add_nms(mod, Keys.task1_rpn_nms, params,
                            task1_rpn_iou_thresh,
                            params.get(Keys.filter_thresh),
                            task1_rpn_max_output_size)
    mod.replace_scope(params.get(Keys.task1_rpn_nms), task1_rpn_nms,
                      get_scope_func, ["task_1/Reshape", "task_1/Reshape_1"])
    task1_rpn_merge = add_merge(mod, Keys.task1_rpn_merge)
    mod.node_add_behind(task1_rpn_nms, task1_rpn_merge)
    task1_rpn_slice = mod.add_new_node(Keys.task1_rpn_slice, "Slice",
                                       {"T": (AT.DTYPE, "float32"),
                                        "Index": (AT.DTYPE, "int32")})
    mod.node_add_behind(task1_rpn_merge, task1_rpn_slice)
    task1_rpn_slice.set_input_node(1,
                                   [rpn_slice_begin.name, rpn_slice_end.name])

    task1_rpn_trans = mod.add_new_node("task1_rpn_trans", "Transpose",
                                       {"Tperm": (AT.DTYPE, "int32"),
                                        "T": (AT.DTYPE, "float32")})
    task1_rpn_reshape = mod.add_new_node("task1_rpn_reshape", "Reshape",
                                         {"T": (AT.DTYPE, "float32"),
                                          "Tshape": (AT.DTYPE, "int32")})
    mod.node_add_behind(task1_rpn_slice, task1_rpn_trans)
    mod.node_add_behind(task1_rpn_trans, task1_rpn_reshape)
    task1_rpn_trans.set_input_node(1, [rpn_trans_idx.name])
    task1_rpn_reshape.set_input_node(1, [rpn_reshape_shape.name])


def replace_rpn(mod: BaseGraph, params: dict):
    rpn_max_output = mod.get_node("task_0/generate_rpn_proposals/"
                                  "non_max_suppression/"
                                  "NonMaxSuppressionV2/"
                                  "max_output_size")
    rpn_max_output_size = rpn_max_output.const_value
    rpn_slice_begin = mod.add_const_node(Keys.rpn_slice_begin,
                                         np.array([0, 0, 0, 0]).astype(np.int32))
    rpn_slice_end = mod.add_const_node(Keys.rpn_slice_end,
                                       np.array([1, 4, 1, rpn_max_output_size]).astype(
                                           np.int32))

    rpn_trans_idx = mod.add_const_node("rpn_trans_idx",
                                       np.array([0, 2, 3, 1]).astype(np.int32))
    rpn_reshape_shape = mod.add_const_node("rpn_reshape_shape",
                                           np.array([-1, 4]).astype(np.int32))
    replace_task0_rpn(mod, params, rpn_slice_begin, rpn_slice_end,
                      rpn_trans_idx, rpn_reshape_shape)
    replace_task1_rpn(mod, params, rpn_slice_begin, rpn_slice_end,
                      rpn_trans_idx, rpn_reshape_shape)

    # decode后数据处理
    trans_decode_res(mod)

    reshape_decode_res(mod, params)

    mod.node_remove_from(params.get(Keys.del_shape_node))

    return True


def add_roi(mod, node_name, params, scale_ratio):
    return mod.add_new_node(node_name, "ROI_NOAH",
                            {"roi_end_mode": (AT.INT, 1),
                             "spatial_scale": (AT.FLOAT, scale_ratio),
                             "sample_num": (AT.INT, 2),
                             "pooled_height": (
                             AT.INT, params.get(Keys.pool_height)),
                             "pooled_width": (
                             AT.INT, params.get(Keys.pool_width))})


def add_scores_slice(mod: BaseGraph):
    task0_fast_slice = mod.add_new_node("task0_fast_slice", "Slice",
                                        {"T": (AT.DTYPE, "float32"),
                                         "Index": (AT.DTYPE, "int32")})
    task1_fast_slice = mod.add_new_node("task1_fast_slice", "Slice",
                                        {"T": (AT.DTYPE, "float32"),
                                         "Index": (AT.DTYPE, "int32")})

    return task0_fast_slice, task1_fast_slice


def add_scores_transpose(mod: BaseGraph):
    task0_fast_scores_trans = mod.add_new_node("task0_fast_scores_trans",
                                               "Transpose",
                                               {"Tperm": (AT.DTYPE, "int32"),
                                                "T": (AT.DTYPE, "float32")})
    task1_fast_scores_trans = mod.add_new_node("task1_fast_scores_trans",
                                               "Transpose",
                                               {"Tperm": (AT.DTYPE, "int32"),
                                                "T": (AT.DTYPE, "float32")})

    return task0_fast_scores_trans, task1_fast_scores_trans


def add_scores_reshape(mod: BaseGraph):
    task0_fast_scores_reshape = mod.add_new_node("task0_fast_scores_reshape",
                                                 "Reshape",
                                                 {"Tshape":
                                                  (AT.DTYPE, "int32"),
                                                  "T": (AT.DTYPE, "float32")})
    task1_fast_scores_reshape = mod.add_new_node("task1_fast_scores_reshape",
                                                 "Reshape",
                                                 {"Tshape":
                                                  (AT.DTYPE, "int32"),
                                                  "T": (AT.DTYPE, "float32")})

    return task0_fast_scores_reshape, task1_fast_scores_reshape


def replace_fast_nms(mod: BaseGraph, params: dict):
    task0_fast_max_output = mod.get_node("task_0/fastrcnn_predictions/map/"
                                         "while/non_max_suppression/"
                                         "NonMaxSuppressionV2/max_output_size")
    task0_fast_max_output_size = task0_fast_max_output.const_value
    task0_fast_iou = mod.get_node("task_0/fastrcnn_predictions/map/while/"
                                  "non_max_suppression/iou_threshold")
    task0_fast_iou_thresh = task0_fast_iou.const_value
    task0_nms_node = add_nms(mod, Keys.task_0_nms, params,
                             task0_fast_iou_thresh,
                             params.get(Keys.score_thresh),
                             task0_fast_max_output_size)
    task0_merge_node = add_merge(mod, Keys.task0_merge)
    mod.replace_scope(params.get(Keys.task_0_nms), task0_nms_node,
                      get_scope_func, [Keys.task_0_clip_boxes,
                                       "task_0/fastrcnn_all_probs"])
    task0_merge_node.set_input_node(0, [task0_nms_node.name])
    task1_fast_max_output = mod.get_node("task_1/fastrcnn_predictions/map/"
                                         "while/non_max_suppression/"
                                         "NonMaxSuppressionV2/max_output_size")
    task1_fast_max_output_size = task1_fast_max_output.const_value
    task1_fast_iou = mod.get_node("task_1/fastrcnn_predictions/map/while/"
                                  "non_max_suppression/iou_threshold")
    task1_fast_iou_thresh = task1_fast_iou.const_value
    task1_nms_node = add_nms(mod, Keys.task_1_nms, params,
                             task1_fast_iou_thresh,
                             params.get(Keys.score_thresh),
                             task1_fast_max_output_size)
    task1_merge_node = add_merge(mod, Keys.task1_merge)
    mod.replace_scope(params.get(Keys.task_1_nms), task1_nms_node,
                      get_scope_func, [Keys.task_1_clip_boxes,
                                       "task_1/fastrcnn_all_probs"])
    task1_merge_node.set_input_node(0, [task1_nms_node.name])


def replace_fast_rcnn(mod: BaseGraph, params: dict):
    out_nodes_set = mod.get_net_output_nodes()
    mod.node_remove(out_nodes_set)
    fast_max_output = mod.get_node("task_0/fastrcnn_predictions/map/while/"
                                   "non_max_suppression/"
                                   "NonMaxSuppressionV2/max_output_size")
    fast_max_output_size = fast_max_output.const_value
    replace_fast_nms(mod, params)

    # slice
    fast_slice_begin = mod.add_const_node("fast_slice_begin",
                                          np.array([0, 1]).astype(np.int32))
    task0_fast_slice_end = mod.add_const_node("task0_fast_slice_end",
                                              np.array([96, 3]).astype(
                                                  np.int32))
    task1_fast_slice_end = mod.add_const_node("task1_fast_slice_end",
                                              np.array([96, 5]).astype(
                                                  np.int32))
    task0_fast_slice, task1_fast_slice = add_scores_slice(mod)

    # transpose
    fast_scores_trans_idx = mod.add_const_node("fast_scores_trans_idx",
                                               np.array([1, 0]).astype(
                                                   np.int32))
    task0_fast_scores_trans, task1_fast_scores_trans = add_scores_transpose(mod)

    # reshape
    fast_scores_shape = mod.add_const_node("fast_scores_shape",
                                           np.array(
                                               [params.get(Keys.batch), 1, -1,
                                                fast_max_output_size]).astype(
                                               np.int32))
    task0_fast_scores_reshape, task1_fast_scores_reshape = add_scores_reshape(mod)

    mod.node_add_behind("task_0/fastrcnn_all_probs", task0_fast_slice.name)
    task0_fast_slice.set_input_node(1, [fast_slice_begin.name,
                                        task0_fast_slice_end.name])
    mod.node_add_behind(task0_fast_slice.name, task0_fast_scores_trans.name)
    task0_fast_scores_trans.set_input_node(1, [fast_scores_trans_idx])
    mod.node_add_behind(task0_fast_scores_trans.name,
                        task0_fast_scores_reshape.name)
    task0_fast_scores_reshape.set_input_node(1, [fast_scores_shape])
    mod.node_add_behind("task_1/fastrcnn_all_probs", task1_fast_slice.name)
    task1_fast_slice.set_input_node(1, [fast_slice_begin.name,
                                        task1_fast_slice_end.name])
    mod.node_add_behind(task1_fast_slice.name, task1_fast_scores_trans.name)
    task1_fast_scores_trans.set_input_node(1, [fast_scores_trans_idx])
    mod.node_add_behind(task1_fast_scores_trans.name,
                        task1_fast_scores_reshape.name)
    task1_fast_scores_reshape.set_input_node(1, [fast_scores_shape])
    return True


def replace_roi(mod: BaseGraph, params: dict):
    adjust_type(mod)
    task0_mul_y = mod.get_node("task_0/mul/y")
    task0_roi_0_ratio = task0_mul_y.const_value
    task0_roi_0 = add_roi(mod, Keys.task_0_roi_0, params,
                          task0_roi_0_ratio[0])
    mod.replace_scope(params.get(Keys.task_0_roi_0), task0_roi_0,
                      get_scope_func,
                      ["conv6/output", "task_0/mul"])
    task0_mul1_y = mod.get_node("task_0/mul_1/y")
    task0_roi_1_ratio = task0_mul1_y.const_value
    task0_roi_1 = add_roi(mod, Keys.task_0_roi_1, params,
                          task0_roi_1_ratio[0])
    mod.replace_scope(params.get(Keys.task_0_roi_1), task0_roi_1,
                      get_scope_func,
                      ["group2_3/Relu", "task_0/mul_1"])
    task1_mul_y = mod.get_node("task_1/mul/y")
    task1_roi_0_ratio = task1_mul_y.const_value
    task1_roi_0 = add_roi(mod, Keys.task_1_roi_0, params,
                          task1_roi_0_ratio[0])
    mod.replace_scope(params.get(Keys.task_1_roi_0), task1_roi_0,
                      get_scope_func,
                      ["conv6/output", "task_1/mul"])
    return True


def gen_noah_third_mod(mod: BaseGraph, new_mod_path: str, mod_params: dict):
    if (not set_inputs(mod, mod_params)):
        return False

    if (not replace_decode(mod, mod_params)):
        return False

    if (not replace_clip_boxes(mod, mod_params)):
        return False

    if (not replace_rpn(mod, mod_params)):
        return False

    if (not replace_fast_rcnn(mod, mod_params)):
        return False

    if (not replace_roi(mod, mod_params)):
        return False

    mod.save_new_model(new_mod_path)

    return True


def gen_om_params(confs: dict):
    b_atc = (confs.get("om_gen_bin") == "atc")
    om_params = confs.get("atc_params") if b_atc else confs.get("omg_params")
    return om_params


def gen_noah(mod_path: str, out_dir: str, confs: dict):
    m_paths, mod_fmk = mod_param_uti.get_third_party_mod_info(mod_path)
    if (m_paths is None):
        return False
    if (mod_fmk not in Support.framework):
        mlog("{} not in support list:{}".format(mod_fmk, Support.framework))
        return False

    atc_params = confs.get("atc_params")
    valid_om_conf = (om_uti.check_support(confs, Support.om_gen_bin) and
                     om_uti.check_support(atc_params, Support.atc_params))
    if (not valid_om_conf):
        return False

    mod_params = confs.get("mod_params")
    if (not check_mod_params(mod_params)):
        return False

    mod = mu.get_mod(m_paths, mod_fmk)

    m_name, m_ext = os.path.splitext(os.path.basename(m_paths[0]))
    new_mod_path = os.path.join(out_dir, "{}__tmp__new{}".format(
        m_name, m_ext))
    if (not gen_noah_third_mod(mod, new_mod_path, mod_params)):
        return False

    om_params = gen_om_params(confs)
    om_path = os.path.join(out_dir, m_name)
    om_uti.gen_om(new_mod_path, "", mod_fmk, om_path,
                  confs.get("om_gen_bin"), om_params)

    if (not confs.get("debug")):
        os.remove(new_mod_path)
    return True
