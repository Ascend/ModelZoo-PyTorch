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
from ...mod_uti import node_uti
from ...mod_uti.log_uti import mlog


class Keys(object):
    in_img_name = "input_img_name"
    in_anchor_name = "input_anchor_name"
    in_bev_anchor_name = "input_bev_anchor_name"
    in_img_anchor_name = "input_img_anchor_name"
    in_calib_p2_name = "input_calib_p2_name"
    in_img_h = "input_img_h"
    in_img_w = "input_img_w"
    rpn_nms_name = "rpn_nms_name"
    anchor_nums = "anchor_nums"
    avod_proj_img_scope = "img_proj_scope"
    nms_type = "NonMaxSuppressionV3"
    proposals_box_ind_name = "proposals_box_ind_name"
    avod_box_ind_name = "avod_box_ind_name"
    calib_coef = "calib_p2_coefficient"


class Support(object):
    # 3rdmod releated
    framework = (mod_param_uti.ModFMK.TF, )
    # om releated
    om_gen_bin = {"om_gen_bin": ("atc", )}
    atc_params = {"soc_version": ("Ascend610", )}   # Ascend710 to verify

    # mod releated
    pad_conv_fixed = False


def check_mod_params(params: dict):
    check_items = {Keys.in_img_name: {"dtype": str, "min": 0},
                   Keys.in_anchor_name: {"dtype": str, "min": 0},
                   Keys.in_bev_anchor_name: {"dtype": str, "min": 0},
                   Keys.in_img_anchor_name: {"dtype": str, "min": 0},
                   Keys.in_calib_p2_name: {"dtype": str, "min": 0},
                   Keys.in_img_h: {"dtype": int, "min": 0},
                   Keys.in_img_w: {"dtype": int, "min": 0},
                   Keys.rpn_nms_name: {"dtype": str, "min": 0},
                   Keys.anchor_nums: {"dtype": (list, tuple),
                                      "min": 0, "max": 2},
                   Keys.proposals_box_ind_name: {"dtype": str, "min": 0},
                   Keys.avod_box_ind_name: {"dtype": str, "min": 0},
                   Keys.avod_proj_img_scope: {"dtype": str, "min": 0},
                   Keys.calib_coef: {"dtype": float, "min": 0.0}}
    if (not mod_param_uti.check_params_dtype_len(params, check_items)):
        return False
    if (not mod_param_uti.check_pos_values(
            Keys.anchor_nums, params.get(Keys.anchor_nums), (int,))):
        return False
    return True


def check_mod_inputs(mod: BaseGraph, params: dict):
    in_img = mod.get_node(params.get(Keys.in_img_name))
    in_anchor = mod.get_node(params.get(Keys.in_anchor_name))
    in_bev_anchor = mod.get_node(params.get(Keys.in_bev_anchor_name))
    in_img_anchor = mod.get_node(params.get(Keys.in_img_anchor_name))
    in_calib = mod.get_node(params.get(Keys.in_calib_p2_name))
    img_valid = node_uti.check_node_fix_shape(in_img, 3, 2, 3)
    anchor_valid = node_uti.check_node_fix_shape(in_anchor, 2, 1, 6)
    bev_anchor_valid = node_uti.check_node_fix_shape(in_bev_anchor, 2, 1, 4)
    img_anchor_valid = node_uti.check_node_fix_shape(in_img_anchor, 2, 1, 4)
    calib_p2_valid = (in_calib.shape == np.array([3, 4], dtype=np.int32)).all()
    return (img_valid and anchor_valid and calib_p2_valid and
            bev_anchor_valid and img_anchor_valid)


def set_inputs(mod: BaseGraph, params: dict):
    input_img_node = mod.get_node(params.get(Keys.in_img_name))
    in_img_h = params.get(Keys.in_img_h)
    in_img_w = params.get(Keys.in_img_w)
    input_img_node.set_shape([in_img_h, in_img_w, 3])
    mlog("set {} shape:({}, {}, 3)".format(Keys.in_img_name,
                                           in_img_h, in_img_w),
         level=logging.INFO)

    anchor_nums = params.get(Keys.anchor_nums)
    if (len(anchor_nums) == 1):
        in_anchor = mod.get_node(params.get(Keys.in_anchor_name))
        in_anchor.set_shape((anchor_nums[0], 6))
        in_bev_anchor = mod.get_node(params.get(Keys.in_bev_anchor_name))
        in_img_anchor = mod.get_node(params.get(Keys.in_img_anchor_name))
        in_bev_anchor.set_shape((anchor_nums[0], 4))
        in_img_anchor.set_shape((anchor_nums[0], 4))
        mlog("set {}\n{}\n{} anchor num:{}".format(
            Keys.in_anchor_name, Keys.in_bev_anchor_name,
            Keys.in_img_anchor_name, anchor_nums[0]), level=logging.INFO)

    in_calib_name = params.get(Keys.in_calib_p2_name)
    coef = params.get(Keys.calib_coef)
    calib_coef_node = mod.add_const_node(in_calib_name + "_coef",
                                         np.array([coef], dtype=np.float32))
    calib_mul_node = mod.add_new_node(in_calib_name + "_mul",
                                      "Mul", {"T": (AT.DTYPE, "float32")})
    mod.node_add_behind(in_calib_name, calib_mul_node)
    calib_mul_node.set_input_node(1, [calib_coef_node])
    mlog("add Mul {} after {}".format(coef, in_calib_name), level=logging.INFO)
    return True


def set_reshape(mod: BaseGraph, params: dict):
    rpn_nms_node = mod.get_node(params.get(Keys.rpn_nms_name))
    rpn_nms_out = mod.get_node(rpn_nms_node.input_name[2]).const_value
    valid_nms_out = (rpn_nms_out.size == 1 and rpn_nms_out > 0)
    if (not valid_nms_out):
        mlog("invalid rpn nms outsize:{}".format(rpn_nms_out))
        return False
    mlog("get rpn nms outsize:{}".format(rpn_nms_out), level=logging.INFO)

    tar_prefix = params.get(Keys.avod_proj_img_scope)
    all_reshape = mod.get_nodes_by_optype("Reshape")
    shape_pt3d = []   # (1, -1)
    shape_pt2d = []   # (-1, 8)
    for node in all_reshape:
        if (tar_prefix not in node.name):
            continue
        shape_node = mod.get_node(node.input_name[1])
        if ((shape_node.const_value == np.array([1, -1])).all()):
            shape_pt3d.append(shape_node)
        elif ((shape_node.const_value == np.array([-1, 8])).all()):
            shape_pt2d.append(shape_node)
        else:
            mlog("unknown avod img proj reshape:{}".format(
                shape_node.const_value))
            return False
    valid_shapes = (len(shape_pt3d) == 3 and len(shape_pt2d) == 4)
    if (not valid_shapes):
        mlog("unexpected shape num. 3d:{}, 2d:{}".format(
            len(shape_pt3d), len(shape_pt2d)))
        return False

    for node in shape_pt3d:
        node.set_value(np.array([1, rpn_nms_out * 8], dtype=np.int32))
    mlog("set avod img proj shape3d to:(1, {} * 8)".format(rpn_nms_out),
         level=logging.INFO)
    for node in shape_pt2d:
        node.set_value(np.array([rpn_nms_out, 8], dtype=np.int32))
    mlog("set avod img proj shape2d to:({}, 8)".format(rpn_nms_out),
         level=logging.INFO)
    return True


def replace_box_ind(mod: BaseGraph, params: dict):
    anchor_num = params.get(Keys.anchor_nums)[0]
    rpn_nms_node = mod.get_node(params.get(Keys.rpn_nms_name))
    rpn_max_out = mod.get_node(rpn_nms_node.input_name[2]).const_value

    box_name_nums = {params.get(Keys.proposals_box_ind_name): anchor_num,
                     params.get(Keys.avod_box_ind_name): rpn_max_out}
    for name, num in box_name_nums.items():
        ori_ind_node = mod.get_node(name)
        new_ind_node = mod.add_const_node(
            ori_ind_node.name + "_box_ind", np.zeros((num, ), dtype=np.int32))
        mod.node_replace(ori_ind_node, new_ind_node,
                         input_node=[], if_remove=False)
        mod.node_remove_from(ori_ind_node)
        mlog("replace {} node with {} and remove other related nodes".format(
            ori_ind_node.name, new_ind_node.name), level=logging.INFO)
    return True


def adjust_nms(mod: BaseGraph, params: dict):
    nms_nodes = mod.get_nodes_by_optype(Keys.nms_type)
    if (len(nms_nodes) != 2):
        mlog("unexpected {} num {}.".format(Keys.nms_type, len(nms_nodes)))
        return False

    for cur_nms in nms_nodes:
        max_out = mod.get_node(cur_nms.input_name[2]).const_value
        iou_th = mod.get_node(cur_nms.input_name[3]).const_value
        score_th = mod.get_node(cur_nms.input_name[4]).const_value
        valid_size = (max_out.size == 1 and iou_th.size == 1 and
                      score_th.size == 1)
        if (not valid_size):
            mlog("unexpected {} max_out:{}, iou:{}, score:{}".format(
                 cur_nms.name, max_out, iou_th, score_th))
            return False

        iou_th = 0 if (iou_th == -np.inf) else iou_th
        score_th = -65504.0 if (score_th == -np.inf) else score_th
        valid_value = (max_out > 0 and iou_th >= 0 and (score_th >= 0 or
                                                        score_th == -65504.0))
        if (not valid_value):
            mlog("unexpected {} max_out:{}, iou:{}, score:{}".format(
                 cur_nms.name, max_out, iou_th, score_th))
            return False

        perm_node = mod.add_const_node(cur_nms.name + "_box_perm",
                                       np.array([1, 0], dtype=np.int32))
        trans_node = mod.add_new_node(cur_nms.name + "_box_trans", "Transpose",
                                      {"Tperm": (AT.DTYPE, "int32")})
        box_node = mod.get_node(cur_nms.input_name[0])
        trans_node.set_input_node(0, (box_node, perm_node))

        new_attr = {"max_output_size": (AT.INT, max_out),
                    "iou_threshold": (AT.FLOAT, iou_th),
                    "score_threshold": (AT.FLOAT, score_th)}
        new_nms = mod.add_new_node(cur_nms.name + "_w_idx",
                                   "NonMaxSuppressionIndex", new_attr)

        score_node = mod.get_node(cur_nms.input_name[1])
        mod.node_replace(cur_nms, new_nms, (trans_node, score_node))
        mod.node_remove(cur_nms.input_name[2:])

        mlog("replace {} with {}".format(cur_nms.name, new_nms.name),
             level=logging.INFO)
    return True


def gen_avod_third_mod(mod: BaseGraph, new_mod_path: str, mod_params: dict):
    if (not set_inputs(mod, mod_params)):
        return False

    if (not set_reshape(mod, mod_params)):
        return False

    if (not replace_box_ind(mod, mod_params)):
        return False

    if (not adjust_nms(mod, mod_params)):
        return False

    mod.save_new_model(new_mod_path)
    return True


def _get_pad_names_before_conv(mod: BaseGraph):
    if (Support.pad_conv_fixed):
        return ()

    res = []
    pad_nodes = mod.get_nodes_by_optype("Pad")
    in_out_map = mod.get_net_in_out_map()
    for node in pad_nodes:
        cur_out = in_out_map.get(node.name)
        out_op_types = [mod.get_node(name).op_type for name in cur_out]
        if ("Conv2D" in out_op_types):
            res.append(node.name)
    return res


def gen_om_params(mod: BaseGraph, mod_params: dict, confs: dict):
    anchor_nums = mod_params.get(Keys.anchor_nums)

    b_atc = (confs.get("om_gen_bin") == "atc")
    om_params = confs.get("atc_params") if b_atc else confs.get("omg_params")
    if (b_atc and len(anchor_nums) > 1):
        dynamic = {"input_format": "ND",
                   "dynamic_dims": "\"",
                   "input_shape": "\"{}:-1,6; {}:-1,4; {}:-1,4\"".format(
                    mod_params.get(Keys.in_anchor_name),
                    mod_params.get(Keys.in_bev_anchor_name),
                    mod_params.get(Keys.in_img_anchor_name))}

        for idx, num in enumerate(anchor_nums):
            end = "" if (idx == (len(anchor_nums) - 1)) else ";"
            dynamic["dynamic_dims"] = "{}{},{},{}{}".format(
                dynamic["dynamic_dims"], num, num, num, end)
        dynamic["dynamic_dims"] = "{}\"".format(dynamic["dynamic_dims"])

        om_params.update(dynamic)
        mlog("set anchor num to dynamic. nums:{}. fmt:ND".format(anchor_nums),
             level=logging.INFO)

    net_out_nodes = mod.get_net_output_nodes()
    net_out_names = ["{}:0".format(node.name) for node in net_out_nodes]
    tar_pad_names = _get_pad_names_before_conv(mod)
    net_out_names += ["{}:0".format(name) for name in tar_pad_names]
    om_params.update({"out_nodes": "{}".format(";".join(net_out_names))})
    mlog("set out nodes:{}.".format(om_params.get("out_nodes")),
         level=logging.INFO)
    return om_params


def gen_avod(mod_path: str, out_dir: str, confs: dict):
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
    if (not check_mod_inputs(mod, mod_params)):
        return False

    m_name, m_ext = os.path.splitext(os.path.basename(m_paths[0]))
    new_mod_path = os.path.join(out_dir, "{}__tmp__new{}".format(
        m_name, m_ext))
    if (not gen_avod_third_mod(mod, new_mod_path, mod_params)):
        return False

    om_params = gen_om_params(mod, mod_params, confs)
    om_path = os.path.join(out_dir, m_name)
    om_uti.gen_om(new_mod_path, "", mod_fmk, om_path,
                  confs.get("om_gen_bin"), om_params)

    if (not confs.get("debug")):
        os.remove(new_mod_path)
    return True
