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

from ...mod_modify.interface import BaseNode
from ...mod_modify.interface import AttrType as AT
from ...mod_uti import mod_uti as mu
from ...mod_uti import mod_param_uti
from ...mod_uti import om_uti
from ...mod_uti.log_uti import mlog
from ...mod_modify.tf_runner import TFRunner


def check_mod_params(params: dict):
    check_items = {Keys.in_img_name: {"dtype": str, "min": 0},
                   Keys.in_shape: {"dtype": (list, tuple), "min": 0},
                   Keys.preprocess_name: {"dtype": str, "min": 0},
                   Keys.first_stage_normalization_ratio: {"dtype": float,
                                                          "min": 0},
                   Keys.first_stage_nms_score_threshold: {"dtype": float},
                   Keys.first_stage_nms_iou_threshold: {"dtype": float,
                                                        "min": 0},
                   Keys.first_stage_max_proposals: {"dtype": int, "min": 0}}
    if (not mod_param_uti.check_params_dtype_len(params, check_items)):
        return False
    return True


def check_second_mod_params(params: dict):
    check_items = {Keys.num_classes: {"dtype": int, "min": 0},
                   Keys.second_stage_decode_scope_name: {"dtype": str,
                                                         "min": 0},
                   Keys.second_stage_nms_scope_name: {"dtype": str, "min": 0},
                   Keys.second_stage_slice_name: {"dtype": str, "min": 0},
                   Keys.second_stage_normalization_ratio: {"dtype": float,
                                                           "min": 0},
                   Keys.second_stage_nms_score_threshold: {"dtype": float,
                                                           "min": 0},
                   Keys.second_stage_nms_iou_threshold: {"dtype": float,
                                                         "min": 0},
                   Keys.second_stage_max_detections_per_class: {"dtype": int,
                                                                "min": 0}}
    if (not mod_param_uti.check_params_dtype_len(params, check_items)):
        return False

    return True


class Keys(object):
    in_img_name = "input_img_name"
    in_shape = "input_shape"
    preprocess_name = "preprocess_name"
    first_stage_normalization_ratio = "first_stage_normalization_ratio"
    first_stage_nms_score_threshold = "first_stage_nms_score_threshold"
    first_stage_nms_iou_threshold = "first_stage_nms_iou_threshold"
    first_stage_max_proposals = "first_stage_max_proposals"
    num_classes = "num_classes"
    second_stage_decode_scope_name = "second_stage_decode_scope_name"
    second_stage_nms_scope_name = "second_stage_nms_scope_name"
    second_stage_slice_name = "second_stage_slice_name"
    second_stage_normalization_ratio = "second_stage_normalization_ratio"
    second_stage_nms_score_threshold = "second_stage_nms_score_threshold"
    second_stage_nms_iou_threshold = "second_stage_nms_iou_threshold"
    second_stage_max_detections_per_class = \
        "second_stage_max_detections_per_class"


class GSupport(object):
    # 3rdmod releated
    framework = (mod_param_uti.ModFMK.TF,)
    # om releated
    om_gen_bin = {"om_gen_bin": ("omg",)}
    omg_params = {"input_format": ("NHWC",)}

    # mod releated
    relu6_conv_fixed = False


class ProcessScope:

    @staticmethod
    def _get_scope_func(node, scope):
        return node.name.startswith(scope + '/')

    @staticmethod
    def _find_in_set(node_name, forward_set):
        for forward_name in forward_set:
            if forward_name.startswith(node_name):
                return True
        return False

    @staticmethod
    def born_decode(mod, node_name, scale_factor1, scale_factor2,
                    scale_factor3, scale_factor4):
        return mod.add_new_node(node_name, "Decode",
                                {"scale_factor1": (AT.FLOAT, scale_factor1),
                                 "scale_factor2": (AT.FLOAT, scale_factor2),
                                 "scale_factor3": (AT.FLOAT, scale_factor3),
                                 "scale_factor4": (AT.FLOAT, scale_factor4)})

    @staticmethod
    def process_cropandresize(mod):
        crop_and_resize_list = mod.get_nodes_by_optype("CropAndResize")
        del_crop_list = []
        for crop_and_resize in crop_and_resize_list:
            ori_box_idx = crop_and_resize.input_name[2]
            del_crop_list.append(ori_box_idx)
        return del_crop_list

    @staticmethod
    def add_cropandresize_const(mod, infer_list, node_list, infer_idx):
        for idx, crop_const_node in enumerate(node_list):
            mod.node_remove_from(crop_const_node)
            mod.add_const_node(crop_const_node, infer_list[infer_idx + idx])

    @staticmethod
    def get_scope_input_const(mod, scope_name):
        scope_node_set = mod.get_subgraph_name(ProcessScope._get_scope_func,
                                               scope_name)
        const_dict = {}
        for sub_node_name in scope_node_set:
            sub_node = mod.get_node(sub_node_name)
            if sub_node.op_type == "Const" and "div" in sub_node_name:
                const_dict[sub_node_name] = sub_node.const_value
        attr_list = [0, 0, 0, 0]
        for idx, mem in enumerate(
                sorted(const_dict.items(), key=lambda item: item[0])[:4]):
            attr_list[idx] = mem[1]
        return attr_list

    @staticmethod
    def delete_scope(mod, scope_name):
        delete_node_set = mod.get_subgraph_name(ProcessScope._get_scope_func,
                                                scope_name)
        deletes_input_node = mod.get_subgraph_input_node_name(
            ProcessScope._get_scope_func, scope_name=scope_name,
            subgraph_node_set=delete_node_set).pop()
        deletes_output_set = mod.get_subgraph_output_node_name(
            ProcessScope._get_scope_func, scope_name=scope_name,
            subgraph_node_set=delete_node_set)
        for out_node in deletes_output_set:
            node_expand = mod.get_node(out_node)
            for idx, pre_node in enumerate(mod.get_input_node(node_expand)):
                if pre_node.name in delete_node_set:
                    node_expand.set_input_node(idx, [deletes_input_node])
        mod.node_remove(delete_node_set)

    @staticmethod
    def replace_scope(mod, scope_name, replace_node, input_set=None,
                      output_set=None):
        replaced_node_set = mod.get_subgraph_name(ProcessScope._get_scope_func,
                                                  scope_name)
        if input_set is None:
            replaced_input_set = mod.get_subgraph_input_node_name(
                ProcessScope._get_scope_func, scope_name=scope_name,
                subgraph_node_set=replaced_node_set)
        else:
            replaced_input_set = input_set
        if output_set is None:
            replaced_output_set = mod.get_subgraph_output_node_name(
                ProcessScope._get_scope_func, scope_name=scope_name,
                subgraph_node_set=replaced_node_set)
        else:
            replaced_output_set = output_set

        for idx, input_node in enumerate(replaced_input_set):
            replace_node.set_input_node(idx, [input_node])

        for out_node in replaced_output_set:
            node_expand = mod.get_node(out_node)
            for idx, pre_node in enumerate(
                    mod.get_input_node_name(node_expand)):
                if pre_node in replaced_node_set:
                    node_expand.set_input_node(idx, [replace_node])
        mod.node_remove(replaced_node_set)

    @staticmethod
    def process_nms(mod, nms_node_name, max_output_size, score_thresh,
                    iou_threshold, clip_y_min, clip_x_min, clip_y_max,
                    clip_x_max, zoom_ratio):
        nms_input_node_list = mod.get_input_node_name(nms_node_name)
        nms_output_node_list = mod.get_output_node_name(nms_node_name)
        change_coord_0, new_nms, merge_proposal, split_node, clip_to_window \
            = ProcessScope.born_nms_node(
                mod, clip_y_max, clip_x_max, clip_y_min, clip_x_min, zoom_ratio,
                max_output_size, iou_threshold, score_thresh)
        for out_node in nms_output_node_list:
            node_expand = mod.get_node(out_node)
            for idx, pre_node in enumerate(mod.get_input_node(node_expand)):
                if pre_node.name == nms_node_name:
                    node_expand.set_input_node(idx, [split_node])

        softmax_node = None
        softmax_begin_node = mod.get_node("Softmax")
        decode_begin_node = mod.get_node("Decode_replace")
        softmax_branch = mod.get_nodes_behind_node(softmax_begin_node,
                                                   end_nodes=[nms_node_name])
        decode_branch = mod.get_nodes_behind_node(decode_begin_node,
                                                  end_nodes=[nms_node_name])
        for idx, input_node in enumerate(nms_input_node_list):
            if input_node in softmax_branch:
                softmax_node = input_node
            elif input_node in decode_branch:
                clip_to_window.set_input_node(0, [input_node])
                change_coord_0.set_input_node(0, [clip_to_window])
            else:
                mod.node_remove_from(input_node)

        new_nms.set_input_node(0, [change_coord_0, softmax_node])
        merge_proposal.set_input_node(0, [new_nms])

        mod.node_remove({nms_node_name})
        return new_nms, clip_to_window, change_coord_0, split_node

    @staticmethod
    def born_nms_node(mod, clip_y_max, clip_x_max, clip_y_min, clip_x_min,
                      zoom_ratio, max_output_size, iou_threshold,
                      score_thresh):
        change_coord_0 = mod.add_new_node("ChangeCoord_0", "ChangeCoord", {
            "y_scale": (AT.FLOAT, clip_y_max / zoom_ratio),
            "x_scale": (AT.FLOAT, clip_x_max / zoom_ratio),
            "bias": (AT.INT, 0), })

        new_nms = mod.add_new_node(
            "FirstStageBatchMultiClassNonMaxSuppression",
            "BatchMultiClassNonMaxSuppression",
            {"max_output_size": (AT.INT, max_output_size),
             "iou_threshold": (AT.FLOAT, iou_threshold),
             "score_threshold": (AT.FLOAT, score_thresh), })
        merge_proposal = mod.add_new_node("MergeProposal", "MergeProposal",
                                          {"start_index": (AT.INT, 0)})

        split_node = mod.add_new_node("FirstStage/GetCoord", "SplitV",
                                      {"num_split": (AT.INT, 2),
                                       "T": (AT.DTYPE, "float32"),
                                       "Tlen": (AT.DTYPE, "int32")})
        split_const_size = mod.add_const_node("first_stage/size_split",
                                              np.array([4, 2], np.int32))
        split_const_dim = mod.add_const_node("first_stage/split_dim",
                                             np.array([3], np.int32))
        split_node.set_input_node(0, [merge_proposal, split_const_size,
                                      split_const_dim])
        clip_to_window = mod.add_new_node(
            "FirstStage/ClipToWindow", "ClipToWindow", {
                "clip_y_min": (
                    AT.FLOAT, clip_y_min), "clip_x_min": (
                    AT.FLOAT, clip_x_min), "clip_y_max": (
                    AT.FLOAT, clip_y_max), "clip_x_max": (
                        AT.FLOAT, clip_x_max)})
        return change_coord_0, new_nms, merge_proposal, split_node, \
            clip_to_window

    @staticmethod
    def process_decode(mod, decode_node, params: dict):
        batch = params.get(Keys.in_shape)[0]
        decode_input_node_list = mod.get_input_node_name(decode_node)
        node_name = ""
        clip_to_window_fn = mod.get_subgraph_name(ProcessScope._get_scope_func,
                                                  "ClipToWindow").pop()
        clip_to_window_branch = mod.get_nodes_behind_node(clip_to_window_fn,
                                                          end_nodes=[
                                                              decode_node])
        for idx, decode_forward_node in enumerate(decode_input_node_list):
            if decode_forward_node in clip_to_window_branch:
                node_name = decode_forward_node
                decode_node.set_input_node(1, [decode_forward_node])

            else:
                ProcessScope._process_decode_const_branch(mod,
                                                          decode_forward_node,
                                                          batch)
                decode_node.set_input_node(0, [decode_forward_node])

        return node_name

    @staticmethod
    def _process_decode_const_branch(mod, decode_forward_node, batch):
        reshape_inputs_list = mod.get_input_node(decode_forward_node)
        for ri_node in reshape_inputs_list:
            if ri_node.op_type == "Const":
                ri_node.set_const_value(np.array([batch, 1, -1, 4], np.int32))

    @staticmethod
    def process_second_decode(mod, decode_node):
        decode_input_node_list = mod.get_input_node_name(decode_node)
        node_name = ""

        map_1_fn = mod.get_subgraph_name(ProcessScope._get_scope_func,
                                         "map_1").pop()

        map_1_branch = mod.get_nodes_behind_node(map_1_fn,
                                                 end_nodes=[decode_node])

        trans_node = mod.add_new_node("user/add/decode/transpose", "Transpose",
                                      {"T": (AT.DTYPE, "float32"),
                                       "Tperm": (AT.DTYPE, "int32")})
        const_node = mod.add_const_node("user/add/decode/transpose/perm",
                                        np.array([0, 1, 3, 2], "int32"))

        for idx, decode_forward_node in enumerate(decode_input_node_list):
            if decode_forward_node in map_1_branch:
                node_name = decode_forward_node
            else:
                decode_node.set_input_node(0, [decode_forward_node])
        trans_node.set_input_node(0, [node_name, const_node])
        decode_node.set_input_node(1, [trans_node])

        return node_name

    @staticmethod
    def process_second_nms(mod, nms_dict, mod_type, nms_node_name,
                           second_softmax_name, max_output_size, score_thresh,
                           iou_threshold, clip_y_min, clip_x_min, clip_y_max,
                           clip_x_max, normal_ratio):
        nms_input_node_list = mod.get_input_node_name(nms_node_name)
        new_nms, trans_node, const_node, trans_node2, const2_node, \
            second_changecoord, clip_to_window, split_end_node, \
            second_changecoord_name, merge_proposal = \
            ProcessScope.born_second_nms_node(
                mod, max_output_size, iou_threshold, score_thresh, clip_y_max,
                clip_x_max, clip_y_min, clip_x_min, normal_ratio)
        for nms_node, nms_shape in nms_dict.items():
            node_expand = mod.get_node(nms_node)
            for idx, pre_node in enumerate(mod.get_input_node(node_expand)):
                if pre_node.name == nms_node_name:
                    if nms_shape[-1] == 4:
                        node_expand.set_input_node(idx, [split_end_node], [0])
                    else:
                        node_expand.set_input_node(idx, [split_end_node], [2])
        del_list = ['detection_boxes', 'detection_scores', 'num_detections',
                    'detection_classes']
        del_nouse_set = set()

        softmax_node = None
        softmax_begin_node = mod.get_node(second_softmax_name)
        decode_begin_node = mod.get_node("second_decode_replace")
        softmax_branch = mod.get_nodes_behind_node(softmax_begin_node,
                                                   end_nodes=[nms_node_name])
        decode_branch = mod.get_nodes_behind_node(decode_begin_node,
                                                  end_nodes=[nms_node_name])
        for idx, input_node in enumerate(nms_input_node_list):
            if input_node in softmax_branch:
                softmax_node = input_node
            elif input_node in decode_branch:
                trans_node.set_input_node(0, [input_node, const_node])
            else:
                mod.node_remove_from(input_node)

        clip_to_window.set_input_node(0, [trans_node])
        second_changecoord.set_input_node(0, [clip_to_window])
        new_nms.set_input_node(0, [second_changecoord_name, trans_node2])
        trans_node2.set_input_node(0, [softmax_node, const2_node])
        merge_proposal.set_input_node(0, [new_nms])
        mod.node_remove({nms_node_name})
        if mod_type == "faster_rcnn":
            for del_node in del_list:
                del_nouse_set = del_nouse_set | mod.get_nodes_forward_node(
                    del_node, end_nodes=[merge_proposal])
            mod.node_remove(del_nouse_set - {merge_proposal.name})
        else:
            for del_node in del_list:
                mod.node_remove_from(del_node)
        return softmax_node

    @staticmethod
    def born_second_nms_node(mod, max_output_size, iou_threshold, score_thresh,
                             clip_y_max, clip_x_max, clip_y_min, clip_x_min,
                             normal_ratio):
        new_nms = mod.add_new_node("BatchMultiClassNonMaxSuppression_1",
                                   "BatchMultiClassNonMaxSuppression", {
                                       "max_output_size": (
                                           AT.INT, max_output_size),
                                       "iou_threshold": (
                                           AT.FLOAT, iou_threshold),
                                       "score_threshold": (
                                           AT.FLOAT, score_thresh), })
        merge_proposal = mod.add_new_node("SecondStage/MergeProposal",
                                          "MergeProposal",
                                          {"start_index": (AT.INT, 1)})

        split_end_node = mod.add_new_node("SecondStage/Split", "SplitV",
                                          {"num_split": (AT.INT, 3),
                                           "T": (AT.DTYPE, "float32"),
                                           "Tlen": (AT.DTYPE, "int32")})
        split_end_const_size = mod.add_const_node("SecondStage/size_split",
                                                  np.array([4, 1, 1],
                                                           np.int32))
        split_end_const_dim = mod.add_const_node("SecondStage/split_dim",
                                                 np.array([3], np.int32))
        split_end_node.set_input_node(0, [merge_proposal, split_end_const_size,
                                          split_end_const_dim])
        trans_node = mod.add_new_node("user/add/clip/transpose", "Transpose",
                                      {"T": (AT.DTYPE, "float32"),
                                       "Tperm": (AT.DTYPE, "int32")})
        const_node = mod.add_const_node("user/add/clip/transpose/perm",
                                        np.array([0, 3, 2, 1], "int32"))

        trans_node2 = mod.add_new_node("user/add/clip/transpose02",
                                       "Transpose",
                                       {"T": (AT.DTYPE, "float32"),
                                        "Tperm": (AT.DTYPE, "int32")})
        const2_node = mod.add_const_node("user/add/clip/transpose/perm02",
                                         np.array([0, 3, 2, 1], "int32"))

        second_changecoord_name = "second/changecoord01"
        second_changecoord = mod.add_new_node(
            second_changecoord_name,
            "ChangeCoord", {
                "y_scale": (AT.FLOAT, clip_y_max / normal_ratio),
                "x_scale": (AT.FLOAT, clip_x_max / normal_ratio),
                "bias": (AT.INT, 0)})

        clip_to_window = mod.add_new_node("clip_to_window", "ClipToWindow", {
            "clip_y_min": (AT.FLOAT, clip_y_min),
            "clip_x_min": (AT.FLOAT, clip_x_min),
            "clip_y_max": (AT.FLOAT, clip_y_max),
            "clip_x_max": (AT.FLOAT, clip_x_max)})
        return new_nms, trans_node, const_node, trans_node2, const2_node, \
            second_changecoord, clip_to_window, split_end_node, \
            second_changecoord_name, merge_proposal

    @staticmethod
    def before_decode_const(mod, const_name, infer_result, infer_idx):
        mod.node_remove_from(const_name)
        decode_const = (infer_result)[infer_idx].transpose(1, 0)
        decode_const = np.expand_dims(decode_const, 0)
        decode_const = np.expand_dims(decode_const, 2)

        mod.add_const_node(const_name, decode_const, if_array=True)

    @staticmethod
    def find_spacetobatch_identity(mod, cur_type):
        infer_list = []
        del_set = set()
        spacetobatch_node_list = mod.get_nodes_by_optype(cur_type)
        for spacetobatch in spacetobatch_node_list:
            input_node_list = mod.get_input_node(spacetobatch)
            for input_node in input_node_list:
                if input_node.op_type == "Identity":
                    infer_list.append(input_node.name)
                    del_set.add(input_node.name)
        return infer_list, del_set

    @staticmethod
    def identity2const(mod, del_set, infer_list, infer_result, infer_idx):
        for node_name in del_set:
            mod.node_remove_from(node_name)

        for const_idx, const_name in enumerate(infer_list):
            value_const = infer_result[infer_idx + const_idx]
            mod.add_const_node(const_name, value_const)

    @staticmethod
    def process_softmax(mod, nms_node, softmax_name, batch):
        reshape_node = mod.get_input_node(softmax_name).pop()
        softmax_node = mod.get_node(softmax_name)

        reshape_const = mod.add_const_node("reshape_const",
                                           np.array([-1, 2], dtype=np.int32))
        reshape_input_list = mod.get_input_node(reshape_node)

        reshape_node.set_input_node(1, [reshape_const])
        mod.node_remove_from(reshape_input_list[1])

        stridedslice_node = mod.add_new_node("SoftmaxStridedSlice",
                                             "StridedSlice",
                                             {"shrink_axis_mask": (AT.INT, 0),
                                              "begin_mask": (AT.INT, 0),
                                              "ellipsis_mask": (AT.INT, 0),
                                              "new_axis_mask": (AT.INT, 0),
                                              "end_mask": (AT.INT, 1),
                                              "T": (AT.DTYPE, "float32"),
                                              "Index": (AT.DTYPE, "int32"), })
        begin_node = mod.add_const_node("SoftmaxStridedSlice/begin",
                                        np.array([0, 1], np.int32))
        end_node = mod.add_const_node("SoftmaxStridedSlice/end",
                                      np.array([0, 2], np.int32))
        strides_node = mod.add_const_node("SoftmaxStridedSlice/strides",
                                          np.array([1, 1], np.int32))
        stridedslice_node.set_input_node(0,
                                         [softmax_node, begin_node, end_node,
                                          strides_node])

        expend_node = mod.get_input_node(nms_node)[1]
        del_set = mod.get_nodes_forward_node(expend_node)
        save_set = mod.get_nodes_forward_node(softmax_name)
        dele_set = del_set - save_set
        mod.node_remove(dele_set)
        softmax_reshape_node = mod.add_new_node("FirstStage/SoftmaxReshape",
                                                "Reshape",
                                                {"T": (AT.DTYPE, "float32"),
                                                 "Tshape": (
                                                     AT.DTYPE, "int32"), })
        softmax_reshape_const = mod.add_const_node("FirstStage/reshape_const",
                                                   np.array([batch, 1, -1, 1],
                                                            np.int32))
        softmax_reshape_node.set_input_node(0, [stridedslice_node,
                                                softmax_reshape_const])
        nms_node.set_input_node(1, [softmax_reshape_node])

    @staticmethod
    def add_trans_from_clip_to_change(mod, clip_node, changecoord_node):
        clip_node = mod.get_node(clip_node)
        changecoord_node = mod.get_node(changecoord_node)

        trans_node = mod.add_new_node("FirstStageTranspose", "Transpose",
                                      {"T": (AT.DTYPE, "float32"),
                                       "Tperm": (AT.DTYPE, "int32")})
        tran_const = mod.add_const_node("FirstSageTranspose/perm",
                                        np.array([0, 1, 3, 2], np.int32))
        trans_node.set_input_node(0, [clip_node, tran_const])
        changecoord_node.set_input_node(0, [trans_node])

    @staticmethod
    def change_const_value(mod, node_name, input_node_shape):
        input_set = mod.get_input_node_name(node_name)
        for input_node_name in input_set:
            input_node = mod.get_node(input_node_name)

            if input_node.op_type == "Const":
                input_node.set_const_value(input_node_shape)


class ThirdStage:
    @staticmethod
    def process_toint32(mod):
        mlog("[ThirdStage]: Processing Toint32 Node.", level=logging.INFO)
        in_out_map = mod.get_net_in_out_map()
        toint_node = mod.get_node("ToInt32")
        reshape_node = mod.get_input_node(toint_node).pop()
        # for merge_proposal add -1
        add_minu_const = mod.add_const_node("add_minu_const",
                                            np.array([-1, ], np.int32),
                                            if_array=False)
        add_minu_node = mod.add_new_node("add_negative", "Add", {})
        add_minu_node.set_input_node(0,
                                     [reshape_node.name, add_minu_const.name])

        after_toint_node = mod.get_node(in_out_map.get(toint_node.name)[0])
        if after_toint_node.op_type != "Add":
            mlog("Please check the node after ToInt32 node, which should be "
                 "add but"
                 " now is {}".format(after_toint_node.op_type))

        add_const = (set(mod.get_input_node_name(after_toint_node)) - {
            toint_node.name}).pop()
        cast_add_node = mod.add_new_node("CastAdd", "CastAdd", {})
        cast_add_node.set_input_node(0, [add_minu_node.name, add_const])
        latest_node_list = in_out_map.get(after_toint_node.name)
        for latest_node in latest_node_list:
            for idx, input_node in enumerate(
                    mod.get_input_node_name(latest_node)):
                if input_node == after_toint_node.name:
                    latest = mod.get_node(latest_node)
                    latest.set_input_node(idx, [cast_add_node.name])
        mod.node_remove({toint_node.name, after_toint_node.name})
        return add_const

    @staticmethod
    def born_add_const(mod, add_const_name, infer_result, infer_idx):
        if infer_idx == 1:
            mod.node_remove_from(add_const_name)
            mod.add_const_node(add_const_name, infer_result[infer_idx])

    @staticmethod
    def find_series_nodes(mod):
        transpose_node_list = mod.get_nodes_by_optype("Transpose")
        mlog("[ThirdStage]: Processing Transpose+Expenddims+Squeeze series "
             "node.", level=logging.INFO)
        for transpose_node in transpose_node_list:
            for transpose_output in mod.get_output_node(transpose_node):
                ThirdStage.find_expand_squeeze(mod, transpose_output,
                                               transpose_node)

    @staticmethod
    def find_expand_squeeze(mod, transpose_output, transpose_node):
        if transpose_output.op_type == "ExpandDims":
            for expenddims_output in mod.get_output_node(transpose_output):
                if expenddims_output.op_type == "Squeeze":
                    ThirdStage.process_trans_exp_squ(mod, transpose_node,
                                                     transpose_output,
                                                     expenddims_output)

    @staticmethod
    def delete_series_node(mod, start_point, end_point, start_inc: bool,
                           end_inc: bool):
        end_forward = mod.get_nodes_forward_node(end_point, if_self=end_inc)
        start_forward = mod.get_nodes_forward_node(start_point,
                                                   if_self=not start_inc)
        delete_set = end_forward - start_forward
        mod.node_remove(delete_set)

    @staticmethod
    def process_trans_exp_squ(mod, trans_node, exp_node, squ_node):
        for squ_output in mod.get_output_node(squ_node):
            squ_output.set_input_node(0, [trans_node])
        exp_input = set(mod.get_input_node_name(exp_node))
        squ_input = set(mod.get_input_node_name(squ_node))
        remove_set = (exp_input | squ_input | {exp_node.name,
                                               squ_node.name}) - {
            trans_node.name}
        mod.node_remove(remove_set)

    @staticmethod
    def custom_process_stride20(mod):
        mlog("[ThirdStage]: Processing Stride20 which shape has changed.",
             level=logging.INFO)
        mod.add_const_node("strided_slice_20_stack",
                           np.array([2], dtype=np.int32))
        mod.add_const_node("strided_slice_20_stack_1",
                           np.array([3], dtype=np.int32))
        strided_slice20 = mod.get_node("strided_slice_20")
        strided_slice20.set_input_node(1, ["strided_slice_20_stack",
                                           "strided_slice_20_stack_1"])
        mod.node_remove(["strided_slice_20/stack", "strided_slice_20/stack_1"])


class ModProcess:
    def __init__(self, mod_type, mod_param, second_mod_param):
        self.mod_type = mod_type
        self.mod_param = mod_param
        self.second_mod_param = second_mod_param

    def _initial_bias(self, mod):
        bias_list = mod.get_nodes_by_optype("BiasAdd")
        for bias_node_name in bias_list:
            bias_node = mod.get_node(bias_node_name)
            format = bias_node.get_attr("data_format", AT.BYTE)
            if not format:
                bias_node.set_attr(
                    {"data_format": (AT.BYTE, bytes("NHWC", "utf-8"))})
        return True

    def final_process_identity(self, mod):
        end_node_list = mod.get_net_output_nodes()
        for end_node in end_node_list:
            if end_node.op_type == "Identity":
                mod.node_remove({end_node.name})
        return True

    def _initial_input(self, mod):
        input_image_node = mod.get_node(self.mod_param.get(Keys.in_img_name))
        if input_image_node.get_attr("dtype", AT.DTYPE) == "uint8":
            cast_node_list = mod.get_output_node(input_image_node)
            if len(cast_node_list) != 1:
                mlog("only support one input:{}".format(len(cast_node_list)))
                return False
            else:
                cast_node = cast_node_list[0]
                casts_output_node_list = mod.get_output_node(cast_node)
                self._initial_input_cast(mod, casts_output_node_list,
                                         cast_node, input_image_node)
                mod.node_remove({cast_node.name})
        input_image_node.set_attr({"shape": (AT.SHAPE, self.preprocess_shape),
                                   "dtype": (AT.DTYPE, "float32")})
        return True

    def _initial_input_cast(self, mod, casts_output_node_list, cast_node,
                            input_image_node):
        for casts_output_node in casts_output_node_list:
            for idx, pre_node in enumerate(
                    mod.get_input_node(casts_output_node)):
                if pre_node == cast_node:
                    casts_output_node.set_input_node(idx, [input_image_node])

    def _del_assert(self, mod):
        mlog("[Initial]: Deleting Assert Node in model.", level=logging.INFO)
        assert_list = mod.get_nodes_by_optype("Assert")
        for assert_node in assert_list:
            mod.node_remove_from(assert_node)
            assert_output_node_list = mod.get_output_node_name(
                assert_node.name)
            for output_node_name in assert_output_node_list:
                old_output_node = mod.get_node(output_node_name)
                old_output_node.remove_input('^' + assert_node.name)
        return True

    def _initial_depthwiseconv2d(self, mod):
        replace_list = mod.get_nodes_by_optype("DepthwiseConv2dNative")

        for idx, replace_node in enumerate(replace_list):
            delete_set = set()

            # 获取filter的最后一个维度，确定需要几份conv
            input_list = mod.get_input_node(replace_node)
            output_list = mod.get_output_node(replace_node)
            delete_set.add(replace_node.name)
            if_process, ori_input, filter_array = self._get_depthwise_const(
                mod, input_list, delete_set)
            # get const shape's channel
            if if_process:
                need_split = filter_array.shape[-2]
                split_name = "tensor_split" + str(idx)
                split_const_name = split_name + "/split_dim"
                concat_name = "tensor_concat" + str(idx)
                concat_const_name = concat_name + "/axis"

                split_node = mod.add_new_node(split_name, "Split", {
                    "num_split": (AT.INT, need_split),
                    "T": (AT.DTYPE, "float32")})
                split_const_node = mod.add_const_node(split_const_name,
                                                      np.array([3],
                                                               dtype=np.int32),
                                                      if_array=False)
                split_node.set_input_node(0, [split_const_node.name,
                                              ori_input.name])
                self._born_concat_node(mod, concat_name, concat_const_name,
                                       need_split, filter_array, idx,
                                       replace_node, split_node, output_list)
                mod.node_remove(delete_set)
        return True

    def _get_depthwise_const(self, mod, input_list, delete_set):
        if_process = True
        ori_input = None
        for input_node in input_list:
            if input_node.op_type == "Identity":
                i_node = mod.get_input_node(input_node)[0]
                filter_array = i_node.const_value
                if filter_array.shape[-1] == 1:
                    if_process = False
                else:
                    delete_set.add(input_node.name)
                    delete_set.add(i_node.name)
            elif input_node.op_type == "Const":
                filter_array = input_node.const_value
                if filter_array.shape[-1] == 1:
                    if_process = False
                else:
                    delete_set.add(input_node.name)
            else:
                ori_input = input_node
        return if_process, ori_input, filter_array

    def _born_concat_node(self, mod, concat_name, concat_const_name,
                          need_split, filter_array, idx, replace_node,
                          split_node, output_list):
        concat_node = mod.add_new_node(concat_name, "ConcatV2",
                                       {"N": (AT.INT, need_split),
                                        "T": (AT.DTYPE, "float32"),
                                        "Tidx": (AT.DTYPE, "int32")})
        concat_const_node = mod.add_const_node(concat_const_name,
                                               np.array([3], dtype=np.int32),
                                               if_array=False)

        for split in range(need_split):
            splited_filter = filter_array[:, :, split:split + 1, :]
            depthwise_name = "depthwiseconv2d" + str(idx) + "_" + str(split)
            depthwise_weight_name = depthwise_name + "_weight"
            conv_node = mod.add_new_node(depthwise_name, "Conv2D",
                                         self._get_attrfromdepthwise(
                                             replace_node))
            weight_node = mod.add_const_node(depthwise_weight_name,
                                             splited_filter, if_array=True)
            conv_node.set_input_node(0, [split_node.name], [split])
            conv_node.set_input_node(1, [weight_node.name])
            concat_node.set_input_node(split, [conv_node.name])
        concat_node.set_input_node(need_split, [concat_const_node.name])
        for idx, output_node in enumerate(output_list):
            output_node.set_input_node(idx, [concat_node.name])

    def _get_attrfromdepthwise(self, replaced_node: BaseNode):
        data_format = replaced_node.get_attr("data_format", AT.BYTE)
        strides = replaced_node.get_attr("strides", AT.LIST_INT)
        dilations = replaced_node.get_attr("dilations", AT.LIST_INT)
        padding = replaced_node.get_attr("padding", AT.BYTE)
        T = replaced_node.get_attr("T", AT.DTYPE)

        return {"data_format": (AT.BYTE, data_format),
                "strides": (AT.LIST_INT, strides),
                "dilations": (AT.LIST_INT, dilations),
                "padding": (AT.BYTE, padding), "T": (AT.DTYPE, T)}

    def mod_initial(self, mod):
        mlog("[Initial]: initial start", level=logging.INFO)

        if (not self._initial_bias(mod)):
            return False

        if (not self._initial_input(mod)):
            return False

        if (not self._del_assert(mod)):
            return False

        if (not self._initial_depthwiseconv2d(mod)):
            return False
        mlog("[Initial]: initial success", level=logging.INFO)
        return True

    def mod_infer(self, mod, ori_mod, mod_type):
        mlog("[ModInfer]: Start mod infer", level=logging.INFO)
        runner = TFRunner(-1, False)
        input_placeholder = ori_mod.get_node(
            self.mod_param.get(Keys.in_img_name))
        input_placeholder.set_shape(self.mod_param.get(Keys.in_shape))
        infer_list = [self.mod_param.get(Keys.preprocess_name)]
        self.add_const_node = None
        self.begin_result = 0
        if mod_type == "mask_rcnn":
            ThirdStage.find_series_nodes(mod)
            ThirdStage.custom_process_stride20(mod)
            self.add_const_node = ThirdStage.process_toint32(mod)
            infer_list.append(self.add_const_node)
            self.begin_result = 1

        scale_1, scale_2, scale_3, scale_4 = \
            ProcessScope.get_scope_input_const(mod, "Decode")

        decode_node = ProcessScope.born_decode(mod, "Decode_replace", scale_1,
                                               scale_2, scale_3, scale_4)
        ProcessScope.replace_scope(mod, "Decode", decode_node)
        self.decode_forward_const = ProcessScope.process_decode(mod,
                                                                decode_node,
                                                                self.mod_param)
        self.sp2ba_list, self.del_spba_set = \
            ProcessScope.find_spacetobatch_identity(mod, "SpaceToBatchND")

        self.ba2sp_list, self.del_basp_set = \
            ProcessScope.find_spacetobatch_identity(mod, "BatchToSpaceND")

        self.before_cropandresize_list = ProcessScope.process_cropandresize(
            mod)
        self.second_nms_output_list = list(
            mod.get_subgraph_output_node_name(
                ProcessScope._get_scope_func,
                scope_name=self.second_mod_param.get(
                    Keys.second_stage_nms_scope_name)))
        infer_list.append(self.decode_forward_const)
        infer_list = infer_list + self.sp2ba_list + self.ba2sp_list + \
            self.before_cropandresize_list + self.second_nms_output_list
        self.infer_result = runner.infer(ori_mod, infer_list)
        ThirdStage.born_add_const(mod, self.add_const_node, self.infer_result,
                                  self.begin_result)

        self.nms_dict = {}

        for nms_idx, nms_node_name in enumerate(self.second_nms_output_list):
            self.nms_dict[nms_node_name] = self.infer_result[
                nms_idx + self.begin_result + 2 + len(self.sp2ba_list) + len(
                    self.ba2sp_list) + len(
                    self.before_cropandresize_list)].shape

        self.preprocess_shape = self.infer_result[0].shape
        mlog("[ModInfer]: Infered successful", level=logging.INFO)

        return True

    def first_stage_process(self, mod):
        mlog("[FirstStage]: Start modify", level=logging.INFO)

        pre_batch, pre_height, pre_width, pre_channel = self.preprocess_shape

        ProcessScope.delete_scope(mod, 'Preprocessor')

        nms_replace_node = mod.add_new_node(
            "BatchMultiClassNonMaxSuppression_replace", "NMS", {})
        ProcessScope.replace_scope(mod, "BatchMultiClassNonMaxSuppression",
                                   nms_replace_node)

        map1_node = mod.add_new_node("Map1", "ChangeCoord", {"y_scale": (
            AT.FLOAT, self.mod_param.get(
                Keys.first_stage_normalization_ratio) / pre_height),
            "x_scale": (AT.FLOAT, self.mod_param.get(
                Keys.first_stage_normalization_ratio) / pre_width),
            "bias": (AT.INT, 0)})
        map_node = mod.add_new_node("Map", "ChangeCoord", {"y_scale": (
            AT.FLOAT,
            self.mod_param.get(Keys.first_stage_normalization_ratio)),
            "x_scale": (AT.FLOAT, self.mod_param.get(
                Keys.first_stage_normalization_ratio)), "bias": (AT.INT, 0)})
        ProcessScope.replace_scope(mod, "map", map_node)
        ProcessScope.replace_scope(mod, "map_1", map1_node)
        map_node.remove_input("Tile")
        nms_node, clip_node, change_node, map1_input = \
            ProcessScope.process_nms(
                mod, nms_replace_node.name,
                self.mod_param.get(Keys.first_stage_max_proposals),
                self.mod_param.get(Keys.first_stage_nms_score_threshold),
                self.mod_param.get(Keys.first_stage_nms_iou_threshold), 0.0, 0.0,
                pre_height, pre_width,
                self.mod_param.get(Keys.first_stage_normalization_ratio))
        map1_node.set_input_node(0, [map1_input], [0])
        map1_node.remove_input(1)
        ProcessScope.before_decode_const(mod, self.decode_forward_const,
                                         self.infer_result,
                                         self.begin_result + 1)
        ProcessScope.identity2const(mod, self.del_spba_set, self.sp2ba_list,
                                    self.infer_result, self.begin_result + 2)
        ProcessScope.identity2const(mod, self.del_basp_set, self.ba2sp_list,
                                    self.infer_result,
                                    self.begin_result + 2 + len(
                                        self.sp2ba_list))
        ProcessScope.add_cropandresize_const(mod, self.infer_result,
                                             self.before_cropandresize_list,
                                             self.begin_result + 2 + len(
                                                 self.sp2ba_list) + len(
                                                 self.ba2sp_list))
        ProcessScope.process_softmax(mod, nms_node, "Softmax", pre_batch)
        ProcessScope.add_trans_from_clip_to_change(mod, clip_node, change_node)
        mlog("[FirstStage]: Modified successful", level=logging.INFO)

        return True

    def second_stage_process(self, mod):
        mlog("[SecondStage]: Start modify", level=logging.INFO)
        second_decode_scope = self.second_mod_param.get(
            Keys.second_stage_decode_scope_name)
        second_nms_scope = self.second_mod_param.get(
            Keys.second_stage_nms_scope_name)
        slice_name = self.second_mod_param.get(Keys.second_stage_slice_name)
        first_stage_max_proposals = self.mod_param.get(
            Keys.first_stage_max_proposals)
        num_class = self.second_mod_param.get(Keys.num_classes)

        scale_1, scale_2, scale_3, scale_4 = \
            ProcessScope.get_scope_input_const(mod, second_decode_scope)

        second_decode_node = ProcessScope.born_decode(mod,
                                                      "second_decode_replace",
                                                      scale_1, scale_2,
                                                      scale_3, scale_4)
        ProcessScope.replace_scope(mod, second_decode_scope,
                                   second_decode_node)
        decode_input_name = mod.get_input_node_name(second_decode_node)
        decode_input_2 = ProcessScope.process_second_decode(mod,
                                                            second_decode_node)

        pre_batch, pre_height, pre_width, pre_channel = self.preprocess_shape
        if "/" in second_decode_scope:
            second_softmax_name = "SecondStagePostprocessor/Softmax"
        elif "_1" in second_decode_scope:
            second_softmax_name = "Softmax_1"

        nms_node = mod.add_new_node(
            "BatchMultiClassNonMaxSuppression_1_replace", "NonMaxSuppression",
            {})
        ProcessScope.replace_scope(mod, second_nms_scope, nms_node,
                                   output_set=self.second_nms_output_list)
        softmax_node = ProcessScope.process_second_nms(
            mod, self.nms_dict, self.mod_type, nms_node.name,
            second_softmax_name,
            self.second_mod_param.get(
                Keys.second_stage_max_detections_per_class),
            self.second_mod_param.get(
                Keys.second_stage_nms_score_threshold),
            self.second_mod_param.get(
                Keys.second_stage_nms_iou_threshold),
            0, 0, pre_height, pre_width,
            self.second_mod_param.get(
                Keys.second_stage_normalization_ratio))

        self._second_stage_process_reshape(pre_batch,
                                           first_stage_max_proposals,
                                           num_class, mod, softmax_node,
                                           decode_input_name, decode_input_2,
                                           slice_name)
        mlog("[SecondStage]: Modified successful", level=logging.INFO)
        return True

    def _second_stage_process_reshape(
            self,
            pre_batch,
            first_stage_max_proposals,
            num_class,
            mod,
            softmax_node,
            decode_input_name,
            decode_input_2,
            slice_name):
        reshape13_node_shape = np.array(
            [pre_batch, first_stage_max_proposals, num_class, 1], np.int32)
        ProcessScope.change_const_value(mod, softmax_node,
                                        reshape13_node_shape)

        reshape9_node_shape = np.array(
            [pre_batch, 4, num_class, first_stage_max_proposals], np.int32)
        reshape10_node_shape = np.array(
            [pre_batch, first_stage_max_proposals, num_class, 4], np.int32)

        for node in decode_input_name:
            if node == decode_input_2:
                ProcessScope.change_const_value(mod, node, reshape9_node_shape)
            else:
                ProcessScope.change_const_value(mod, node,
                                                reshape10_node_shape)

        second_const_shape = np.array([num_class + 1], np.int32)
        second_const = mod.add_const_node("second/const01", second_const_shape)

        slice = mod.get_node(slice_name)
        delete_set = mod.get_nodes_forward_node(slice, [1, 2], if_self=True)
        replaced_input_set = mod.get_input_node_name(slice)
        delete_set = delete_set | set(replaced_input_set)

        slice_output_node_list = mod.get_output_node_name(slice)

        for out_node in slice_output_node_list:
            node_expand = mod.get_node(out_node)
            for idx, pre_node in enumerate(mod.get_input_node(node_expand)):
                if pre_node.name == slice_name:
                    node_expand.set_input_node(idx, [second_const])

        mod.node_remove(delete_set)
        return True


def gen_rcnn_mod(mod, ori_mod, new_mod_path: str, mod_type: str,
                 mod_params: dict, second_mod_params: dict):
    # 统计需要推理的部分，统一进行推理

    mod_process = ModProcess(mod_type, mod_params, second_mod_params)
    if not mod_process.mod_infer(mod, ori_mod, mod_type):
        return False
    if (not mod_process.mod_initial(mod)):
        return False
    if (not mod_process.second_stage_process(mod)):
        return False

    if (not mod_process.first_stage_process(mod)):
        return False
    if not mod_process.final_process_identity(mod):
        return False
    mod.save_new_model(new_mod_path)

    return True


def _get_relu6_names_before_conv(mod):
    res = []
    if (GSupport.relu6_conv_fixed):
        return res

    pad_nodes = mod.get_nodes_by_optype("Relu6")
    in_out_map = mod.get_net_in_out_map()
    for node in pad_nodes:
        cur_out = in_out_map.get(node.name)
        out_op_types = [mod.get_node(name).op_type for name in cur_out]
        if ("Conv2D" in out_op_types):
            res.append(node.name)
    return res


def _get_relu_names_before_concat(mod):
    res = []
    if (GSupport.relu6_conv_fixed):
        return res

    pad_nodes = mod.get_nodes_by_optype("Relu")
    in_out_map = mod.get_net_in_out_map()
    concat_set = set()
    for node in pad_nodes:
        cur_out = in_out_map.get(node.name)
        out_op_types = [mod.get_node(name).op_type for name in cur_out]
        if ("ConcatV2" in out_op_types) and not set(cur_out).issubset(
                concat_set) and (node.name).startswith("SecondStage"):
            res.append(node.name)
            concat_set = concat_set | set(cur_out)
    return res


def gen_om_params(mod, confs: dict):
    # 生成模型--指定的参数
    om_params = confs.get("omg_params")

    net_out_nodes = mod.get_net_output_nodes()
    net_out_names = [node.name + ":0" for node in net_out_nodes]
    tar_pad_names = _get_relu6_names_before_conv(mod)
    tar_relu_names = _get_relu_names_before_concat(mod)
    net_out_names += [name + ":0" for name in tar_pad_names]
    net_out_names += [name + ":0" for name in tar_relu_names]
    if "SecondStage/MergeProposal:0" not in net_out_names:
        net_out_names += ["SecondStage/MergeProposal:0", ]
    om_params.update({"out_nodes": "{}".format(";".join(net_out_names))})
    return om_params


def gen_rcnn(mod_path: str, out_dir: str, confs: dict):
    m_paths, mod_fmk = mod_param_uti.get_third_party_mod_info(mod_path)
    if (m_paths is None):
        return False
    if (mod_fmk not in GSupport.framework):
        mlog("{} not in support list:{}".format(mod_fmk, GSupport.framework))
        return False

    omg_params = confs.get("omg_params")
    valid_om_conf = (om_uti.check_support(confs,
                                          GSupport.om_gen_bin) and
                     om_uti.check_support(
        omg_params, GSupport.omg_params))
    if (not valid_om_conf):
        return False
    mod_type = confs.get("mod_type")
    mod_params = confs.get("mod_params")
    if (not check_mod_params(mod_params)):
        return False
    second_mod_params = confs.get("second_stage_mod_params")
    if (not check_second_mod_params(second_mod_params)):
        return False
    mod = mu.get_mod(m_paths, mod_fmk)

    ori_mod = mu.get_mod(m_paths, mod_fmk)

    m_name, m_ext = os.path.splitext(os.path.basename(m_paths[0]))
    new_mod_path = os.path.join(out_dir,
                                "{}__tmp__new{}".format(m_name, m_ext))
    if (not gen_rcnn_mod(mod, ori_mod, new_mod_path, mod_type, mod_params,
                         second_mod_params)):
        return False

    om_params = gen_om_params(mod, confs)
    om_path = os.path.join(out_dir, m_name)
    om_uti.gen_om(new_mod_path, "", mod_fmk, om_path, confs.get("om_gen_bin"),
                  om_params)

    if (not confs.get("debug")):
        os.remove(new_mod_path)
    return True
