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
import onnx
from ...mod_modify.interface import BaseNode
from ...mod_modify.interface import BaseRunner
from ...mod_modify.interface import BaseGraph
from ...mod_modify.onnx_runner import OXRunner
from ...mod_modify.interface import AttrType as AT
from ...mod_uti import mod_uti as mu
from ...mod_uti import mod_param_uti
from ...mod_uti import om_uti
from ...mod_uti.log_uti import mlog

G_ONNX_NP_DTYPE_MAP = {onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16,
                       onnx.TensorProto.INT32: np.int32, onnx.TensorProto.INT64: np.int64,
                       onnx.TensorProto.BOOL: np.bool, onnx.TensorProto.INT16: np.int16,
                       onnx.TensorProto.INT8: np.int8}


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
    second_stage_max_detections_per_class = "second_stage_max_detections_per_class"


class GSupport(object):
    # 3rdmod releated
    framework = (mod_param_uti.ModFMK.ONNX,)
    # om releated
    om_gen_bin = {"om_gen_bin": ("atc",)}
    atc_params = {"soc_version": ("Ascend610",)}

    # mod releated
    constant_collapse = False


class DetrModifer(object):
    # TODO: input,output node get function
    # TODO: the structure of DETR reshape node
    @staticmethod
    def _get_tar_concat(mod: BaseGraph):
        res = []
        for idx, node in enumerate(mod.graph.node):
            if (node.op_type == "Concat"):
                in_nodes = mod.get_net_input_nodes()
                out_nodes = mod.get_net_output_nodes()
                if (in_nodes == ["Unsqueeze"] and out_nodes == ["Reshape"]):
                    res.append((node, idx))
        return res

    # TODO: 替换掉 _get_tar_concat
    @staticmethod
    def _get_tar_reshape_input(mod: BaseGraph):
        res = []
        reshape_node_list = mod.get_nodes_by_optype("Reshape")
        for reshape_node in reshape_node_list:
            input_node_2nd = reshape_node.input_name[1]
            shape_node = mod.get_node(input_node_2nd)
            if shape_node.op_type not in ["Constant", "Initializer"]:
                res.append(shape_node.name)
        return list(set(res))

    @staticmethod
    def _modify_reshape(mod, reshape_nodes, reshape_const):
        assert (len(reshape_nodes) == len(reshape_const))
        in_out_map = mod.get_net_in_out_map()
        for idx, reshape_node_name in enumerate(reshape_nodes):
            reshape_node = mod.get_node(reshape_node_name)
            output_list = in_out_map.get(reshape_node.name)
            mod.node_remove_from(reshape_node, if_self=True)
            const_node = mod.add_const_node("reshape_const_" + str(idx), reshape_const[idx][0])
            for output_name in output_list:
                out_node = mod.get_node(output_name)
                out_node.set_input_node(1, [const_node])
        return True

    @staticmethod
    def _find_matmul_forward_transpose(mod: BaseGraph):
        need_infer_list = []
        # Initializer->slice->transpose->matmul
        mm_nodes = mod.get_nodes_by_optype("Matmul")
        for mm_node in mm_nodes:
            trans_node = mod.get_node(mm_node.input_name[1])
            DetrModifer._find_series_node(mod, trans_node, need_infer_list)
        return need_infer_list

    @staticmethod
    def _find_flatten(mod: BaseGraph):
        flatten_nodes = mod.get_nodes_by_optype("Flatten")

    @staticmethod
    def _find_series_node(mod, trans_node, need_infer_list):
        if trans_node.op_type == "Transpose":
            slice_node = mod.get_node(trans_node.input_name[0])
            if slice_node.op_type == "Slice":
                initial_node = mod.get_node(slice_node.input_name[0])
                if initial_node.op_type == "Initializer":
                    need_infer_list.append(trans_node.name)

    @staticmethod
    def _find_matmul(mod: BaseGraph):
        input1_names = []
        input2_names = []
        matmul_names = []
        matmul_node_list = mod.get_nodes_by_optype("MatMul")
        for idx, matmul_node in enumerate(matmul_node_list):
            input1_names.append(matmul_node.input_name[0])
            input2_names.append(matmul_node.input_name[1])
            matmul_names.append(matmul_node.name)

        return input1_names, input2_names, matmul_names


    @staticmethod
    def process_resize(mod: BaseGraph, resize_nodes, const_list):
        for idx, const_result in enumerate(const_list):
            resize_node = mod.get_node(resize_nodes[idx])
            del_node = resize_node.input_name[3]
            mod.node_remove_from(del_node)
            resize_size_const = mod.add_const_node("Resize_size_" + str(idx), const_result[0])
            resize_node.set_input_node(3, [resize_size_const])
        return True


    @staticmethod
    def born_trans(mod: BaseGraph, idx, input_node, trans_list):
        trans_node = mod.add_new_node("Transpose_" + str(idx), "Transpose",
                                      {"Tperm": (AT.LIST_INT, trans_list)})
        trans_node.set_input_node(0, [input_node])
        return trans_node

    @staticmethod
    def process_insnormal(mod: BaseGraph):
        ins_normals = mod.get_nodes_by_optype("InstanceNormalization")
        in_out_map = mod.get_net_in_out_map()
        for idx, ins_normal in enumerate(ins_normals):
            eps = ins_normal.get_attr("epsilon", AT.FLOAT)
            layernormalize = mod.add_new_node("LayerNormalization_" + str(idx), "LayerNormalize",
                                              {"eps": (AT.FLOAT, eps), "elementwise": (AT.INT, 0),
                                               "dims": (AT.INT, 2)})
            input_list = ins_normal.input_name
            layernormalize.set_input_node(0, input_list)
            out_node = mod.get_node(in_out_map.get(ins_normal.name)[0])
            out_input = out_node.input_name
            out_input[0] = layernormalize
            out_node.set_input_node(0, out_input)
            mod.node_remove_from(ins_normal)
        return True

    @staticmethod
    def process_tranpose(mod: BaseGraph, trans_nodes, trans_result):
        for idx, trans_node_name in enumerate(trans_nodes):
            mod.node_remove_from(trans_node_name)
            mod.add_const_node(trans_node_name, trans_result[idx])
        return True

    @staticmethod
    def process_reshape(mod: BaseGraph, reshape_input, reshape_dict):
        idx = 0
        for reshape_data, value_list in reshape_dict.items():
            for value in value_list:
                reshape_const_name, modi_dim = value
                reshape_const = mod.get_node(reshape_const_name)
                const_value = list(reshape_const.const_value)
                const_value[modi_dim] = reshape_input[idx][0].shape[modi_dim]
                reshape_const.set_const_value(np.array(const_value, np.int32))
            idx += 1
        return True

    @staticmethod
    def process_where(mod: BaseGraph, mask_off):
        in_out_map = mod.get_net_in_out_map()
        where_nodes = mod.get_nodes_by_optype("Where")

        if mask_off == 1:
            for where_node in where_nodes:
                condition_name, x_const_name, y_node_name = where_node.input_name
                y_node = mod.get_node(y_node_name)
                output_list = in_out_map[where_node.name]
                for output_node_name in output_list:
                    output_node = mod.get_node(output_node_name)
                    outs_list = output_node.input_name
                    for idx, outs_input_name in enumerate(outs_list):
                        outs_input_node = mod.get_node(outs_input_name)
                        if outs_input_node.name == where_node.name:
                            outs_list[idx] = y_node
                    output_node.set_input_node(0, outs_list)
                mod.node_remove_from(where_node)
        else:
            for idx, where_node in enumerate(where_nodes):
                cond_node = mod.get_node(where_node.input_name[0])
                if cond_node.op_type != "Cast":
                    cast_node = mod.add_new_node("Cast_cond_" + str(idx), "Cast",
                                                 {"to": (AT.INT, 9)})
                    cast_node.set_input_node(0, [cond_node])
                    where_input_list = where_node.input_name
                    where_input_list[0] = cast_node
                    where_node.set_input_node(0, where_input_list)
        return True

    @staticmethod
    def process_flatten(mod: BaseGraph):
        flatten_nodes = mod.get_nodes_by_optype('Flatten')
        for idx, flatten_node in enumerate(flatten_nodes):
            forward_node = mod.get_node(flatten_node.input_name[0])
            before_cast = mod.add_new_node("flatten_cast_" + str(idx), "Cast", {"to": (AT.INT, 1)})
            before_cast.set_input_node(0, [forward_node])
            flatten_node.set_input_node(0, [before_cast])
        return True

    @staticmethod
    def tile_init_data(initer, axis, tile_n):
        initer = np.expand_dims(initer, axis=0)
        tile_shape = [1 for d in initer.shape]
        tile_shape[axis] = tile_n
        value = np.tile(initer, tile_shape)
        return value

    @staticmethod
    def find_resize_forward_const(mod: BaseGraph):
        resize_nodes = mod.get_nodes_by_optype("Resize")
        need_infer_const = []
        resize_node_list = []
        for resize_node in resize_nodes:
            input_size = resize_node.input_name[3]
            size_node = mod.get_node(input_size)
            if size_node.op_type not in ["Constant", "Initializer"]:
                need_infer_const.append(size_node.name)
                resize_node_list.append(resize_node.name)
        return resize_node_list, need_infer_const

    @staticmethod
    def tile_init_data_42(initer, axis, tile_n):
        np_dtype = G_ONNX_NP_DTYPE_MAP[initer.data_type]
        initer.dims.insert(axis, tile_n)
        tile_shape = [1 for d in initer.dims]
        tile_shape[axis] = tile_n
        value = np.frombuffer(initer.raw_data, dtype=np_dtype)
        value = np.tile(value, tile_shape)
        value = np.expand_dims(value, axis=1)
        initer.raw_data = value.tobytes()

    @staticmethod
    def del_node(mod: BaseGraph, node_name):
        node = mod.get_node(node_name)
        in_out_map = mod.get_net_in_out_map()
        input_node = mod.get_node(node.input_name[0])
        out_node = mod.get_node(in_out_map.get(node.name)[0])
        out_lists = out_node.input_name
        out_lists[0] = input_node
        out_node.set_input_node(0, out_lists)
        removed = mod.node_remove_from(node_name)

    @staticmethod
    def _find_reshape0(mod: BaseGraph):
        reshape_dict = {}
        reshape_list = mod.get_nodes_by_optype("Reshape")
        for reshape_node in reshape_list:
            reshape_input_list = reshape_node.input_name
            input_shape = mod.get_node(reshape_input_list[1])
            if input_shape.op_type == "Constant" or input_shape.op_type == "Initializer":
                shape_value = mod.get_node(reshape_input_list[1]).const_value
                if 0 in shape_value:
                    if reshape_dict.get(reshape_input_list[0]):
                        reshape_dict[reshape_input_list[0]].append(
                            [reshape_input_list[1], list(shape_value).index(0)])
                    else:
                        reshape_dict[reshape_input_list[0]] = [
                            [reshape_input_list[1], list(shape_value).index(0)]]
        return reshape_dict

    @staticmethod
    def process_gather(mod: BaseGraph):
        mlog("Processing gather[indices=-1] op",level=logging.INFO)
        gather_nodes = mod.get_nodes_by_optype("Gather")
        for gather_node in gather_nodes:
            input_nodes = gather_node.input_name
            for input_name in input_nodes:
                input_node = mod.get_node(input_name)
                if input_node.op_type in ["Constant", "Initializer"]:
                    indices = input_node.const_value
                    if indices == [-1]:
                        input_node.set_const_value(np.array([5], np.int32))
        return True

    @staticmethod
    def _find_sstrideslice(mod: BaseGraph):
        # find special strideslice
        special_slice_dict = {}
        slice_nodes = mod.get_nodes_by_optype("Slice")
        for slice_node in slice_nodes:
            if len(slice_node.input_name) == 5:
                input_data, starts, ends, axes, steps = slice_node.input_name

                step_node = mod.get_node(steps)

                step_value = list(step_node.const_value)[0]
                if step_value == 2:
                    starts_node = mod.get_node(starts)
                    axes_node = mod.get_node(axes)
                    starts_value = list(starts_node.const_value)[0]
                    axes_value = list(axes_node.const_value)[0]
                    if special_slice_dict.get(input_data):
                        special_slice_dict[input_data].append(
                            [slice_node, starts_value, axes_value])
                    else:
                        special_slice_dict[input_data] = []
                        special_slice_dict[input_data].append(
                            [slice_node, starts_value, axes_value])
        return special_slice_dict

    @staticmethod
    def mod_infer(mod):
        runner = OXRunner(0, True)
        infer_list = []

        cc_nodes = DetrModifer._get_tar_reshape_input(mod)
        infer_list.extend(cc_nodes)
        cc_len = len(infer_list)
        resize_node_list, need_infer_const = DetrModifer.find_resize_forward_const(mod)
        infer_list.extend(need_infer_const)
        size_len = len(infer_list)
        ss_dict = DetrModifer._find_sstrideslice(mod)
        ss_len = len(infer_list)
        infer_list.extend(list(ss_dict.keys()))
        reshape_dict = DetrModifer._find_reshape0(mod)
        reshape_len = len(infer_list)
        infer_list.extend(list(reshape_dict.keys()))

        dtype_list = ["int64"] * size_len + ["float32"] * (len(infer_list) - size_len)
        infer_result = runner.infer(mod, infer_list, dtype_list)
        mlog("Infer finished",level=logging.INFO)
        return cc_nodes, infer_result[:cc_len], resize_node_list, infer_result[
                                                                  cc_len:size_len], infer_result[
                                                                                    ss_len:reshape_len], ss_dict, infer_result[
                                                                                                                  reshape_len:], reshape_dict


def gen_detr_mod(ori_mod, mod, mask_off, new_mod_path):
    if mask_off not in [0, 1]:
        mlog("Please check your switch of mask{}, which only support 0 (ON) or 1(OFF)".format(
            mask_off))
    cc_nodes, cc_outs, resize_nodes, size_consts, ss_input, ss_dict, reshape_input, reshape_dict \
        = DetrModifer.mod_infer(ori_mod)

    if not DetrModifer.process_gather(mod):
        return False
    if not DetrModifer.process_reshape(mod, reshape_input, reshape_dict):
        return False
    if not DetrModifer.process_resize(mod, resize_nodes, size_consts):
        return False
    if not DetrModifer._modify_reshape(mod, cc_nodes, cc_outs):
        return False
    if not DetrModifer.process_flatten(mod):
        return False
    if not DetrModifer.process_where(mod, mask_off):
        return False
    if not DetrModifer.process_insnormal(mod):
        return False
    mod.save_new_model(new_mod_path)
    return True


def gen_atc_params(confs: dict):
    b_atc = (confs.get("om_gen_bin") == "atc")
    om_params = confs.get("atc_params") if b_atc else confs.get("omg_params")
    return om_params


def gen_detr(mod_path: str, out_dir: str, confs: dict):
    m_paths, mod_fmk = mod_param_uti.get_third_party_mod_info(mod_path)
    if (m_paths is None):
        return False
    if (mod_fmk not in GSupport.framework):
        mlog("{} not in support list:{}".format(mod_fmk, GSupport.framework))
        return False

    atc_params = confs.get("atc_params")
    valid_om_conf = (
            om_uti.check_support(confs, GSupport.om_gen_bin) and om_uti.check_support(atc_params,
                                                                                      GSupport.atc_params))
    if (not valid_om_conf):
        return False

    mod = mu.get_mod(m_paths, mod_fmk)
    ori_mod = mu.get_mod(m_paths, mod_fmk)

    m_name, m_ext = os.path.splitext(os.path.basename(m_paths[0]))
    new_mod_path = os.path.join(out_dir, "{}__tmp__new{}".format(m_name, m_ext))
    if (not gen_detr_mod(ori_mod, mod, confs.get("mask_off"), new_mod_path)):
        return False

    om_params = gen_atc_params(confs)
    om_path = os.path.join(out_dir, m_name)
    om_uti.gen_om(new_mod_path, "", mod_fmk, om_path, confs.get("om_gen_bin"), om_params)
    # if (not confs.get("debug")):
    #     os.remove(new_mod_path)
    return True
