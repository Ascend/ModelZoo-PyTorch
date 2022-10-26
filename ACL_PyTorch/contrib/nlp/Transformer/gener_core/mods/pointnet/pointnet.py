# -*- coding: utf-8 -*-
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
import numpy as np
import tensorflow as tf
import os
from enum import Enum

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2

from ...mod_uti import mod_param_uti
from ...mod_uti import om_uti
from ...mod_uti.log_uti import mlog

CUSTOM_OP = [
    'FarthestPointSample',
    'GatherPoint',
    'QueryBallPoint',
    'GroupPoint',
    'ThreeNN',
    'ThreeInterpolate',
    'PyFunc']
ND_OP = ['Tile', 'Maximum']
NCHW_OP = ['Relu', 'FusedBatchNorm', 'Conv2D', 'BiasAdd', 'Const']
SPECIAL_IN_CUS = ['PyFunc']


class GSupport(object):
    # 3rdmod releated
    framework = (mod_param_uti.ModFMK.TF,)
    # om releated
    om_gen_bin = {"om_gen_bin": ("omg",)}
    omg_params = {"input_format": ("NHWC",)}


class ModifyMode(Enum):
    NO_TRANS = 0
    HWC_TO_NCHW = 1
    NDC_TO_N1DC = 2
    INSERT_DIM = 3
    AMEND_VALUE = 4


class BasicAct:
    def strip_dropout(
            self,
            input_graph,
            drop_scope,
            dropout_before,
            dropout_after):
        input_nodes = input_graph.node
        nodes_after_strip = []
        for node in input_nodes:
            if node.name.startswith(drop_scope + '/'):
                continue

            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            if new_node.name == dropout_after:
                new_input = []
                self._strip_dropout_sub(
                    new_node, drop_scope, dropout_before, new_input)
                del new_node.input[:]
                new_node.input.extend(new_input)
            nodes_after_strip.append(new_node)

        new_gd = graph_pb2.GraphDef()
        new_gd.node.extend(nodes_after_strip)
        return new_gd

    def _strip_dropout_sub(
            self,
            new_node,
            drop_scope,
            dropout_before,
            new_input):
        for input_name in new_node.input:
            if input_name.startswith(drop_scope + '/'):
                new_input.append(dropout_before)
            else:
                new_input.append(input_name)

    def remove_merge(self, old_gd):
        old_nodes = old_gd.node

        names_to_remove = {}
        for node in old_nodes:
            if node.op == 'Merge':
                names_to_remove[node.name] = node.input[0]
                names_to_remove[node.input[1]] = None
        nodes_after_modify = []
        for node in old_nodes:
            if node.name in names_to_remove:
                continue
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            input_before_removal = node.input
            del new_node.input[:]
            for full_input_name in input_before_removal:
                while full_input_name in names_to_remove:
                    full_input_name = names_to_remove[full_input_name]
                new_node.input.append(full_input_name)
            nodes_after_modify.append(new_node)

        new_gd = graph_pb2.GraphDef()
        new_gd.node.extend(nodes_after_modify)
        return new_gd

    def update_pbmodel(self, model_path):
        graph_def = tf.GraphDef()
        with open(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())

        # fix nodes
        for node in graph_def.node:
            self._update_pbmodel_sub(node)

        return graph_def

    def _update_pbmodel_sub(self, node):
        # special process in pb model with  RefSwitch and AssignSub
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    def remove_node(self, node_name, old_graph):
        old_nodes = old_graph.node
        names_to_remove = {}
        for node in old_nodes:
            if node.name != node_name:
                continue
            names_to_remove[node.name] = [
                x for x in node.input if self.get_node_name(old_graph, x).op != "Const"]

        nodes_after_modify = []

        for node in old_nodes:
            if node.name in names_to_remove:
                continue
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            input_before_removal = node.input
            del new_node.input[:]
            self._remove_node_sub(
                input_before_removal,
                names_to_remove,
                new_node)
            nodes_after_modify.append(new_node)

        new_gd = graph_pb2.GraphDef()
        new_gd.node.extend(nodes_after_modify)
        return new_gd

    def _remove_node_sub(
            self,
            input_before_removal,
            names_to_remove,
            new_node):
        for full_input_name in input_before_removal:
            if full_input_name in names_to_remove:
                for input_name in names_to_remove[full_input_name]:
                    new_node.input.append(input_name)
            else:
                new_node.input.append(full_input_name)

    def add_between_node(
            self,
            b_node,
            a_node_list,
            added_node,
            add_const,
            old_gd,
            pb_path):
        old_nodes = old_gd.node
        nodes_after_modify = []
        new_node_list = []
        for node in old_nodes:
            if node.name == b_node:
                new_node_list = self._born_new_node(
                    node.name, added_node, add_const, pb_path)
                if len(new_node_list) > 1:
                    new_node_list[1].input.insert(0, b_node)
                if new_node_list[0] in old_nodes:
                    pass
                else:
                    nodes_after_modify.extend(new_node_list)
            self._add_between_node_sub(
                node, a_node_list, b_node, new_node_list)
            nodes_after_modify.append(node)

        new_gd = graph_pb2.GraphDef()
        new_gd.node.extend(nodes_after_modify)
        return new_gd

    def _add_between_node_sub(self, node, a_node_list, b_node, new_node_list):
        if node.name in a_node_list:
            tmp_list = node.input
            idx = 0
            for i in range(len(tmp_list)):
                if b_node == tmp_list[i]:
                    idx = i
                    break
            node.input[idx] = new_node_list[len(new_node_list) - 1].name

    def _born_new_node(self, node_name, add_name, add_const, pb_path):
        insert_graph = self.update_pbmodel(pb_path)
        cur_node_list = []
        for node in insert_graph.node:
            if node.name == add_name:
                if len(node.input) > 1:
                    del node.input[0]
                    node.input[0] = node_name + '_' + node.input[0]
                    node.name = node_name + '_' + node.name
                else:
                    del node.input[0]
                    node.input.append(node_name)
                    node.name = node_name + '_' + node.name
                cur_node_list.append(node)
            if node.name == add_const:
                node.name = node_name + '_' + node.name
                cur_node_list.append(node)
        return cur_node_list

    def get_node_name(self, graph_def, name):
        for n in graph_def.node:
            if n.name == name:
                return n
        return None

    def modify_node(
            self,
            attr_name,
            old_gd,
            modify_mod=ModifyMode.NO_TRANS,
            attr="value",
            after_wegt=1):
        old_nodes = old_gd.node
        nodes_after_modify = []
        for node in old_nodes:
            new_node = node_def_pb2.NodeDef()
            if attr_name == node.name:
                new_node = self._modify_attr(
                    old_gd, node.name, modify_mod, attr, after_wegt)
            else:
                new_node.CopyFrom(node)

            nodes_after_modify.append(new_node)

        new_gd = graph_pb2.GraphDef()
        new_gd.node.extend(nodes_after_modify)
        return new_gd

    def _modify_attr(self, graph_def, node_name, modify_mod, attr, after_wegt):
        cur_node = self.get_node_name(graph_def, node_name)
        if attr == "value":
            before_node_dim = self.get_node_name(
                graph_def, node_name).attr[attr].tensor
            kernel_weight = tensor_util.MakeNdarray(before_node_dim)
            type = kernel_weight.dtype

            if modify_mod == ModifyMode.HWC_TO_NCHW:
                kernel_weight_list = list(kernel_weight)
                shape = [1] * 4
                idx = 3
                for i in np.array(kernel_weight_list).shape[::-1]:
                    shape[idx] = i
                    idx -= 1
                tmp = shape[3]
                shape[3] = shape[2]
                shape[2] = shape[1]
                shape[1] = tmp
                after_weight = np.array(kernel_weight_list).reshape(shape)

            elif modify_mod == ModifyMode.AMEND_VALUE:
                after_weight = after_wegt

            elif modify_mod == ModifyMode.INSERT_DIM:
                kernel_weight_list = list(kernel_weight)
                kernel_weight_list.insert(1, 1)
                after_weight = np.array(kernel_weight_list)

            elif modify_mod == ModifyMode.NDC_TO_N1DC:
                kernel_weight_list = list(kernel_weight)
                after_weight = np.array(kernel_weight_list).reshape(
                    kernel_weight.shape[0], 1, kernel_weight.shape[1], kernel_weight.shape[2])
            else:
                after_weight = kernel_weight

            cur_node.attr[attr].CopyFrom(
                attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(
                        after_weight, type)))
        return cur_node


class ModifyPointNet:
    def __init__(self, ori_pb, modify_pbname="modified_pb.pb", remove_dp=True):
        self.basic_act = BasicAct()
        old_graph = self.basic_act.update_pbmodel(ori_pb)
        if remove_dp is True:
            self.final_graph = self.basic_act.strip_dropout(
                old_graph,
                drop_scope='dp1/cond',
                dropout_before='conv1d-fc1/Relu',
                dropout_after='conv1d-fc2/conv1d/ExpandDims')
        else:
            self.final_graph = old_graph
        self.modify_pbname = modify_pbname
        self.expand_path = os.path.join(
            os.path.abspath(
                os.path.dirname(
                    os.path.abspath(__file__)) +
                os.path.sep +
                "."),
            'ExpandDims.pb')
        self.pad_path = os.path.join(
            os.path.abspath(
                os.path.dirname(
                    os.path.abspath(__file__)) +
                os.path.sep +
                "."),
            'pad.pb')
        self.squeeze_path = os.path.join(
            os.path.abspath(
                os.path.dirname(
                    os.path.abspath(__file__)) +
                os.path.sep +
                "."),
            'Squeeze.pb')

    def main_compute(self):
        # for model to success generate,inside bug of framework
        final_graph = self.modify_conv(self.final_graph)
        final_graph = self.modify_input_const(final_graph, CUSTOM_OP)
        final_graph = self.truediv_zero_condition(final_graph)

        final_graph = self.remove_squeeze(final_graph, CUSTOM_OP)
        final_graph = self.remove_expanddims(final_graph, CUSTOM_OP)
        final_graph = self.add_exp_before_custom(final_graph, CUSTOM_OP)

        final_graph = self.add_node_before_nd(final_graph, CUSTOM_OP, ND_OP)
        final_graph = self.add_node_before_concat(final_graph)
        final_graph = self.add_node_after_max(final_graph, CUSTOM_OP)
        final_graph = self.tt_block_modify(final_graph)

        final_graph = self.remove_isolate_node(final_graph)
        tf.io.write_graph(final_graph, "./", self.modify_pbname, as_text=False)
        return True

    def modify_input_const(self, old_gd, custom_list):
        old_nodes = old_gd.node
        for node in old_nodes:
            if node.op in custom_list:
                self._modify_input_const_sub(node, old_gd)
        return old_gd

    def _modify_input_const_sub(self, node, old_gd):
        for b_node_name in node.input:
            b_node = self.basic_act.get_node_name(old_gd, b_node_name)
            if b_node is not None and b_node.op == "Const":
                old_gd = self.basic_act.modify_node(
                    b_node.name, old_gd, modify_mod=ModifyMode.HWC_TO_NCHW)

    def remove_squeeze(self, old_gd, custom_list):
        old_nodes = old_gd.node
        for node in old_nodes:
            if node.op in custom_list:
                old_gd = self._remove_squeeze_sub(node, old_gd)

        return old_gd

    def _remove_squeeze_sub(self, node, old_gd):
        for b_node_name in node.input:
            b_node = self.basic_act.get_node_name(old_gd, b_node_name)
            if b_node is not None and b_node.op == 'Squeeze':
                old_gd = self.basic_act.remove_node(b_node.name, old_gd)
        return old_gd

    def in_range_threenn_interp(self, old_gd, three_node, tt_mode):
        old_nodes = old_gd.node

        if three_node.op != "ThreeInterpolate" and three_node.name not in tt_mode:
            tt_mode.append(three_node.name)
            for tt_node in old_nodes:
                if three_node.name in tt_node.input:
                    self.in_range_threenn_interp(old_gd, tt_node, tt_mode)

    def get_tt_list(self, old_gd):
        tt_list = []
        old_nodes = old_gd.node
        for node in old_nodes:
            if node.op == "ThreeNN":
                self.in_range_threenn_interp(old_gd, node, tt_list)
        return tt_list

    def tt_block_modify(self, old_gd):
        tt_list = self.get_tt_list(old_gd)
        for node_name in tt_list:
            tt_node = self.basic_act.get_node_name(old_gd, node_name)
            if tt_node.op == "Tile":
                old_gd = self.basic_act.modify_node(
                    tt_node.name + "/multiples", old_gd, modify_mod=ModifyMode.INSERT_DIM)
            if tt_node.op == "Sum":
                old_gd = self.basic_act.modify_node(
                    tt_node.name +
                    "/reduction_indices",
                    old_gd,
                    modify_mod=ModifyMode.AMEND_VALUE,
                    after_wegt=3)

        return old_gd

    def remove_expanddims(self, old_gd, custom_list):
        old_nodes = old_gd.node
        for node in old_nodes:
            if node.op == 'ExpandDims':
                old_gd = self._remove_expanddims_sub(node, old_gd, custom_list)
        return old_gd

    def _remove_expanddims_sub(self, node, old_gd, custom_list):
        for b_node_name in node.input:
            b_node = self.basic_act.get_node_name(old_gd, b_node_name)
            if b_node is not None and b_node.op in custom_list:
                old_gd = self.basic_act.remove_node(node.name, old_gd)
        return old_gd

    def modify_conv(self, old_gd):
        old_nodes = old_gd.node
        for node in old_nodes:
            if node.op == "Conv2D":
                node_weight_name = '/weights/read'
                for b_node_name in node.input:
                    b_node = self.basic_act.get_node_name(old_gd, b_node_name)
                    old_gd = self._modify_conv_sub(
                        b_node, node_weight_name, old_gd)
        return old_gd

    def _modify_conv_sub(self, b_node, node_weight_name, old_gd):
        if b_node is not None and b_node.op == "ExpandDims":
            for bb_node_name in b_node.input:
                if node_weight_name in bb_node_name:
                    old_gd = self.basic_act.remove_node(b_node.name, old_gd)
                    old_gd = self.basic_act.modify_node("/".join(bb_node_name.split('/')[:-1]),
                                                        old_gd, modify_mod=ModifyMode.NDC_TO_N1DC)
        return old_gd

    def add_node_before_nd(self, old_gd, custom_list, nd_list):
        old_nodes = old_gd.node
        for node in old_nodes:
            if node.op in nd_list:
                old_gd = self._add_node_before_nd_sub(
                    node, old_gd, custom_list)

        return old_gd

    def _add_node_before_nd_sub(self, node, old_gd, custom_list):
        for b_node_name in node.input:
            b_node = self.basic_act.get_node_name(old_gd, b_node_name)
            if b_node is not None and b_node.op in custom_list:
                old_gd = self.basic_act.add_between_node(
                    b_node.name, [node.name], "Pad", "Pad/paddings", old_gd, self.pad_path)
        return old_gd

    def add_node_after_max(self, old_gd, custom_list):
        old_nodes = old_gd.node
        for node in old_nodes:
            if node.op in custom_list:
                old_gd = self._add_node_after_max_sub(node, old_gd)

        return old_gd

    def _add_node_after_max_sub(self, node, old_gd):
        for b_node_name in node.input:
            b_node = self.basic_act.get_node_name(old_gd, b_node_name)
            if b_node is not None and b_node.op == "Max":
                old_gd = self.basic_act.add_between_node(
                    b_node.name, [node.name], "Pad", "Pad/paddings", old_gd, self.pad_path)
        return old_gd

    def add_node_before_concat(self, old_gd):
        """
        只有threeinterp/GatherPoint/GroupPoint需要concat，且原生GroupPoint就是四维，
        因此这里只考虑threeinterp/GatherPoint两种情况
        大部分情况下，ThreeInterpolate的维度要大些，因此考虑转换另一边输入的维度
        而GatherPoint的维度要小写，因此直接转换GatherPoint的维度
        若有其他情况出现，为了优化性能可以进行修改
        """
        old_nodes = old_gd.node
        for node in old_nodes:
            if node.op == 'ConcatV2':
                for b_node_name in node.input:
                    b_node = self.basic_act.get_node_name(old_gd, b_node_name)
                    old_gd = self._add_node_before_concat_sub(
                        b_node, node, old_gd)
        return old_gd

    def _add_node_before_concat_sub(self, b_node, node, old_gd):
        if b_node is not None and b_node.op == "ThreeInterpolate":
            input_list = node.input[:self.get_idx(b_node.name, node.input)] + node.input[
                self.get_idx(
                    b_node.name,
                    node.input) + 1:]
            for real_input_name in input_list:
                b_node = self.basic_act.get_node_name(old_gd, real_input_name)
                if b_node is not None and b_node.op != "Const":
                    old_gd = self.basic_act.add_between_node(
                        b_node.name, [
                            node.name], "ExpandDims", "ExpandDims/dim", old_gd, self.expand_path)
                    old_gd = self.basic_act.add_between_node(b_node.name + '_ExpandDims',
                                                             [node.name], "Pad", "Pad/paddings",
                                                             old_gd, self.pad_path)
            old_gd = self.basic_act.modify_node(
                node.name + "/axis",
                old_gd,
                modify_mod=ModifyMode.AMEND_VALUE,
                after_wegt=3)
        elif b_node is not None and b_node.op == "GatherPoint":
            old_gd = self.basic_act.add_between_node(
                b_node.name, [node.name], "Pad", "Pad/paddings", old_gd, self.pad_path)
            old_gd = self.basic_act.add_between_node(
                b_node.name + '_Pad', [node.name], "Squeeze", "-1", old_gd, self.squeeze_path)
        return old_gd

    def add_exp_before_custom(self, old_gd, custom_list):
        old_nodes = old_gd.node
        before_inside = []
        for node in old_nodes:
            if node.op in custom_list and node.op not in SPECIAL_IN_CUS:
                old_gd = self._add_exp_before_custom_sub(
                    node, old_gd, before_inside, custom_list)
        for b_node in before_inside:
            need_add = []
            for node in old_nodes:
                if node.op in custom_list and b_node in node.input:
                    need_add.append(node.name)
            bnode = self.basic_act.get_node_name(old_gd, b_node)
            if bnode.op == "ConcatV2":
                old_gd = self.basic_act.add_between_node(
                    b_node, need_add, "ExpandDims", "ExpandDims/dim", old_gd, self.expand_path)
                old_gd = self.basic_act.add_between_node(
                    b_node + "_ExpandDims", need_add, "Pad", "Pad/paddings", old_gd, self.pad_path)
            else:
                old_gd = self.basic_act.add_between_node(
                    b_node, need_add, "ExpandDims", "ExpandDims/dim", old_gd, self.expand_path)
        return old_gd

    def _add_exp_before_custom_sub(
            self,
            node,
            old_gd,
            before_inside,
            custom_list):
        for b_node_name in node.input:
            b_node = self.basic_act.get_node_name(old_gd, b_node_name)
            if b_node is not None and b_node.op not in custom_list and b_node_name not in \
                    before_inside and b_node_name not in self.get_tt_list(
                        old_gd) and b_node.op not in NCHW_OP and not self.if_max_fourdim(old_gd,
                                                                                         b_node_name):
                before_inside.append(b_node_name)
        return old_gd

    def if_max_fourdim(self, old_gd, node_name):
        node = self.basic_act.get_node_name(old_gd, node_name)
        if_fourdim = node.attr["keep_dims"].b
        return if_fourdim

    def truediv_zero_condition(self, old_gd):
        old_nodes = old_gd.node
        for node in old_nodes:
            if 'Maximum/y' in node.name:
                before_node_dim = self.basic_act.get_node_name(
                    old_gd, node.name).attr['value'].tensor
                kernel_weight = tensor_util.MakeNdarray(before_node_dim)
                if kernel_weight - 0.00001 < 0:
                    old_gd = self.basic_act.modify_node(
                        node.name, old_gd, modify_mod=ModifyMode.AMEND_VALUE, after_wegt=0.0001)
        return old_gd

    def remove_isolate_node(self, old_gd):
        old_nodes = old_gd.node
        nodes_has_out = []
        nodes_has_noin = []
        for node in old_nodes:
            nodes_has_out.extend([x for x in node.input])
            if len(node.input) == 0:
                nodes_has_noin.append(node.name)
        nodes_has_output = list(set(nodes_has_out))
        nodes_after_modify = []
        for node in old_nodes:
            if node.name not in nodes_has_output and node.name in nodes_has_noin:
                continue
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)

            nodes_after_modify.append(new_node)

        new_gd = graph_pb2.GraphDef()
        new_gd.node.extend(nodes_after_modify)
        return new_gd

    def get_idx(self, node_name, node_list):
        idx = 0
        for nn_name in node_list:
            if nn_name == node_name:
                return idx
            idx += 1
        return 0


def gen_point_net_mod(m_paths, new_mod_path: str):
    # 统计需要推理的部分，统一进行推理

    mod_process = ModifyPointNet(m_paths, new_mod_path)
    if not mod_process.main_compute():
        return False

    return True


def gen_om_params(confs: dict):
    # 生成模型--指定的参数
    om_params = confs.get("omg_params")
    return om_params


def gen_point_net(mod_path: str, out_dir: str, confs: dict):
    m_paths, mod_fmk = mod_param_uti.get_third_party_mod_info(mod_path)
    if (m_paths is None):
        return False
    if (mod_fmk not in GSupport.framework):
        mlog("{} not in support list:{}".format(mod_fmk, GSupport.framework))
        return False

    omg_params = confs.get("omg_params")
    valid_om_conf = (
        om_uti.check_support(
            confs,
            GSupport.om_gen_bin) and om_uti.check_support(
            omg_params,
            GSupport.omg_params))
    if (not valid_om_conf):
        return False

    m_name, m_ext = os.path.splitext(os.path.basename(m_paths[0]))
    new_mod_path = os.path.join(
        out_dir, "{}__tmp__new{}".format(
            m_name, m_ext))
    if (not gen_point_net_mod(m_paths[0], new_mod_path)):
        return False

    om_params = gen_om_params(confs)
    om_path = os.path.join(out_dir, m_name)
    om_uti.gen_om(
        new_mod_path,
        "",
        mod_fmk,
        om_path,
        confs.get("om_gen_bin"),
        om_params)

    if (not confs.get("debug")):
        os.remove(new_mod_path)
    return True
