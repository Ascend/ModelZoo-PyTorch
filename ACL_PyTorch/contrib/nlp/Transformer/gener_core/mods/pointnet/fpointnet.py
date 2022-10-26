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
import os
import numpy as np
import tensorflow as tf
from . import pointnet
from ...mod_modify.tf_graph_nodes import TfGraphNodes
from ...mod_modify.interface import AttrType as AT
from ...mod_uti import mod_param_uti
from ...mod_uti import om_uti
from ...mod_uti.log_uti import mlog

CUSTOM_OP = ['FarthestPointSample', 'GatherPoint', 'QueryBallPoint', 'GroupPoint', 'ThreeNN',
             'ThreeInterpolate', 'PyFunc']
ND_OP = ['Tile', 'Maximum']


class GSupport(object):
    # 3rdmod releated
    framework = (mod_param_uti.ModFMK.TF,)
    # om releated
    om_gen_bin = {"om_gen_bin": ("omg",)}
    omg_params = {"input_format": ("NHWC",)}


class CustomModifyPointNet:
    def __init__(self, ori_pb, remove_dp=True):
        self.basic_act = pointnet.BasicAct()
        self.auto_cut = pointnet.ModifyPointNet(ori_pb)
        old_graph = self.basic_act.update_pbmodel(ori_pb)
        if remove_dp is True:
            self.final_graph = self.basic_act.strip_dropout(old_graph, drop_scope='dp1/cond',
                                                            dropout_before='conv1d-fc1/Relu',
                                                            dropout_after='conv1d-fc2/conv1d/ExpandDims')
        else:
            self.final_graph = old_graph

    def main_modify(self):
        # for model to success generate,inside bug of framework
        expand_path = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."),
            'ExpandDims.pb')
        pad_path = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."),
            'pad.pb')
        final_graph = self.auto_cut.modify_conv(self.final_graph)
        final_graph = self.auto_cut.modify_input_const(final_graph, CUSTOM_OP)
        final_graph = self.auto_cut.truediv_zero_condition(final_graph)

        final_graph = self.auto_cut.remove_squeeze(final_graph, CUSTOM_OP)
        final_graph = self.auto_cut.remove_expanddims(final_graph, CUSTOM_OP)
        final_graph = self.auto_cut.add_exp_before_custom(final_graph, CUSTOM_OP)

        final_graph = self.auto_cut.add_node_before_nd(final_graph, CUSTOM_OP, ND_OP)
        final_graph = self.basic_act.add_between_node('concat_1', ['fa_layer3/concat'],
                                                      "ExpandDims", "ExpandDims/dim", final_graph,
                                                      expand_path)
        final_graph = self.basic_act.add_between_node('concat_1' + '_ExpandDims',
                                                      ['fa_layer3/concat'], "Pad", "Pad/paddings",
                                                      final_graph, pad_path)

        final_graph = self.auto_cut.tt_block_modify(final_graph)

        final_graph = self.auto_cut.remove_isolate_node(final_graph)
        tf.io.write_graph(final_graph, "./", 'tmp.pb', as_text=False)
        final_graph = self.custom_modify()
        return final_graph

    def custom_modify(self):
        model_name = 'tmp.pb'
        graph_sample = TfGraphNodes(model_name)
        del_node = graph_sample.get_node('layer3/ExpandDims')
        graph_sample.node_remove_connect(del_node, 0, if_remove=False)
        graph_sample.node_remove([del_node.input_name[1], del_node])
        del_node = graph_sample.get_node('fa_layer1/ExpandDims')
        graph_sample.node_remove_connect(del_node, 0, if_remove=False)
        graph_sample.node_remove([del_node.input_name[1], del_node])
        del_node = graph_sample.get_node('fa_layer2/ExpandDims')
        graph_sample.node_remove_connect(del_node, 0, if_remove=False)
        graph_sample.node_remove([del_node.input_name[1], del_node])
        del_node = graph_sample.get_node('fa_layer3/ExpandDims')
        graph_sample.node_remove_connect(del_node, 0, if_remove=False)
        graph_sample.node_remove([del_node.input_name[1], del_node])
        del_node = graph_sample.get_node('ssg-layer3/ExpandDims')
        graph_sample.node_remove_connect(del_node, 0, if_remove=False)
        graph_sample.node_remove([del_node.input_name[1], del_node])

        self.replace_max_node(graph_sample, 'layer1/Max', 'layer1/Maxpool', [1, 1, 32, 1])
        self.replace_max_node(graph_sample, 'layer1/Max_1', 'layer1/Maxpool_1', [1, 1, 64, 1])
        self.replace_max_node(graph_sample, 'layer1/Max_2', 'layer1/Maxpool_2', [1, 1, 128, 1])
        self.replace_max_node(graph_sample, 'layer2/Max', 'layer2/Maxpool', [1, 1, 64, 1])
        self.replace_max_node(graph_sample, 'layer2/Max_1', 'layer2/Maxpool_1', [1, 1, 64, 1])
        self.replace_max_node(graph_sample, 'layer2/Max_2', 'layer2/Maxpool_2', [1, 1, 128, 1])
        self.replace_max_node(graph_sample, 'layer3/maxpool', 'layer3/Maxpool', [1, 32, 1, 1])
        self.replace_max_node(graph_sample, 'ssg-layer1/maxpool', 'ssg-layer1/Maxpool',
                              [1, 1, 64, 1])
        self.replace_max_node(graph_sample, 'ssg-layer2/maxpool', 'ssg-layer2/Maxpool',
                              [1, 1, 64, 1])
        self.replace_max_node(graph_sample, 'ssg-layer3/maxpool', 'ssg-layer3/Maxpool',
                              [1, 32, 1, 1])
        self.modify_concatv2(graph_sample, 'layer3/concat/axis')
        self.modify_concatv2(graph_sample, 'fa_layer1/concat/axis')
        self.modify_concatv2(graph_sample, 'fa_layer2/concat/axis')
        self.modify_concatv2(graph_sample, 'fa_layer3/concat/axis')
        self.modify_concatv2(graph_sample, 'ssg-layer3/concat/axis')
        del1_node = graph_sample.get_node('layer1/concat_3_ExpandDims_Pad')
        graph_sample.node_remove_connect(del1_node, 0, if_remove=False)
        graph_sample.node_remove([del1_node.input_name[1], del1_node])

        del_node = graph_sample.get_node('layer1/concat_3_ExpandDims')
        graph_sample.node_remove_connect(del_node, 0, if_remove=False)
        graph_sample.node_remove([del_node.input_name[1], del_node])

        del_node = graph_sample.get_node('ssg-layer2/Squeeze')
        graph_sample.node_remove_connect(del_node, 0, if_remove=True)
        os.remove('tmp.pb')
        return graph_sample

    def replace_max_node(self, graph_sample, old_node_name, new_node_name, ksize_list):
        old_node = graph_sample.get_node(old_node_name)
        new_node = graph_sample.add_new_node(new_node_name, "MaxPool",
                                             {"dtype": (AT.DTYPE, "float32")})
        new_node.set_attr_value("data_format", "NHWC", "string", )
        new_node.set_attr_value("ksize", ksize_list, "int32", if_list=True)
        new_node.set_attr_value("padding", "VALID", "string")
        new_node.set_attr_value("strides", [1, 1, 1, 1], "int32", if_list=True)

        graph_sample.node_replace(old_node, new_node, [old_node.input_name[0]], if_remove=False)
        graph_sample.node_remove([old_node.input_name[1], old_node])

    def modify_concatv2(self, graph_sample, const_name):
        old_node = graph_sample.get_node(const_name)
        old_node.set_value(np.array([-1]).astype(np.int32), False)


def gen_fpoint_net_mod(m_paths, new_mod_path: str):
    # 统计需要推理的部分，统一进行推理
    mod_process = CustomModifyPointNet(m_paths)
    mod = mod_process.main_modify()
    if not mod:
        return False

    mod.save_new_model(new_mod_path)

    return True


def gen_om_params(confs: dict):
    # 生成模型--指定的参数
    om_params = confs.get("omg_params")
    return om_params


def gen_fpoint_net(mod_path: str, out_dir: str, confs: dict):
    m_paths, mod_fmk = mod_param_uti.get_third_party_mod_info(mod_path)
    if (m_paths is None):
        return False
    if (mod_fmk not in GSupport.framework):
        mlog("{} not in support list:{}".format(mod_fmk, GSupport.framework))
        return False

    omg_params = confs.get("omg_params")
    valid_om_conf = (om_uti.check_support(confs, GSupport.om_gen_bin) and om_uti.check_support(
        omg_params, GSupport.omg_params))
    if (not valid_om_conf):
        return False

    m_name, m_ext = os.path.splitext(os.path.basename(m_paths[0]))
    new_mod_path = os.path.join(out_dir, "{}__tmp__new{}".format(m_name, m_ext))
    if (not gen_fpoint_net_mod(m_paths[0], new_mod_path)):
        return False

    om_params = gen_om_params(confs)
    om_path = os.path.join(out_dir, m_name)
    om_uti.gen_om(new_mod_path, "", mod_fmk, om_path, confs.get("om_gen_bin"), om_params)

    if (not confs.get("debug")):
        os.remove(new_mod_path)
    return True
