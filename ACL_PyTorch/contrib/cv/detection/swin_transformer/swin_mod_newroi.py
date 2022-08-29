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

import onnx
import sys

batchsize = sys.argv[1]

onnx_path = '../swin_net_bs{}.onnx'.format(batchsize)
onnx_model = onnx.load(onnx_path)
graph = onnx_model.graph


cast_mod_nodes = ['Add_18446', 'Add_18691', 'Add_18936', 'Mul_19055', 'Add_19181', 'Add_19426', 'Add_19063', 'Add_19075',
                  'Div_19083', 'Clip_19093', 'Div_19091',
                  'Expand_442', 'Expand_426', 'Expand_434', 'Expand_418', 'Expand_530', 'Expand_522', 'Expand_538',
                  'Expand_546', 'Expand_650', 'Expand_642', 'Expand_634', 'Expand_626', 'Expand_754', 'Expand_738',
                  'Expand_746', 'Expand_730', 'Expand_858', 'Expand_850', 'Expand_842', 'Expand_834', 'Expand_962',
                  'Expand_946', 'Expand_938', 'Expand_954', 'Expand_1170', 'Expand_1162', 'Expand_1154', 'Expand_1146',
                  'Expand_1058', 'Expand_1042', 'Expand_1066', 'Expand_1050', 'Expand_1274', 'Expand_1266',
                  'Expand_1258', 'Expand_1250', 'Expand_2027', 'Expand_2019', 'Expand_2011', 'Expand_2003',
                  'Expand_2131', 'Expand_2123', 'Expand_2107', 'Expand_2115', 'Expand_2235', 'Expand_2227',
                  'Expand_2219', 'Expand_2211', 'Expand_2331', 'Expand_2323', 'Expand_2339', 'Expand_2315',
                  'Expand_2443', 'Expand_2435', 'Expand_2427', 'Expand_2419', 'Expand_2547', 'Expand_2523',
                  'Expand_2531', 'Expand_2539', 'Expand_2643', 'Expand_2635', 'Expand_2627', 'Expand_2651',
                  'Expand_2755', 'Expand_2747', 'Expand_2739', 'Expand_2731', 'Expand_2859', 'Expand_2851',
                  'Expand_2843', 'Expand_2835', 'Expand_3604', 'Expand_3596', 'Expand_3588', 'Expand_3612',
                  'Expand_3716', 'Expand_3708', 'Expand_3700', 'Expand_3692', 'Expand_3804', 'Expand_3820',
                  'Expand_3812', 'Expand_3796', 'Expand_3924', 'Expand_3916', 'Expand_3900', 'Expand_3908',
                  'Expand_4028', 'Expand_4020', 'Expand_4012', 'Expand_4004', 'Expand_4124', 'Expand_4116',
                  'Expand_4108', 'Expand_4132', 'Expand_4236', 'Expand_4228', 'Expand_4220', 'Expand_4212',
                  'Expand_4340', 'Expand_4332', 'Expand_4324', 'Expand_4316', 'Expand_4444', 'Expand_4436',
                  'Expand_4428', 'Expand_4420', 'Expand_5083', 'Expand_5067', 'Expand_5075', 'Expand_5059',
                  'Expand_5187', 'Expand_5179', 'Expand_5171', 'Expand_5163', 'Expand_5291', 'Expand_5283',
                  'Expand_5275', 'Expand_5267', 'Expand_5395', 'Expand_5379', 'Expand_5387', 'Expand_5371',
                  'Expand_5491', 'Expand_5499', 'Expand_5483', 'Expand_5475', 'Expand_5603', 'Expand_5595',
                  'Expand_5587', 'Expand_5579', 'Expand_5699', 'Expand_5707', 'Expand_5691', 'Expand_5683',
                  'Expand_5811', 'Expand_5803', 'Expand_5795', 'Expand_5787', 'Expand_5907', 'Expand_5899',
                  'Expand_5891', 'Expand_5915', 'Expand_6546', 'Expand_6538', 'Expand_6554', 'Expand_6530',
                  'Expand_6658', 'Expand_6642', 'Expand_6650', 'Expand_6634', 'Expand_6762', 'Expand_6754',
                  'Expand_6746', 'Expand_6738', 'Expand_6866', 'Expand_6858', 'Expand_6850', 'Expand_6842',
                  'Expand_6970', 'Expand_6962', 'Expand_6954', 'Expand_6946', 'Expand_7074', 'Expand_7066',
                  'Expand_7050', 'Expand_7058', 'Expand_7178', 'Expand_7170', 'Expand_7154', 'Expand_7162',
                  'Expand_7282', 'Expand_7274', 'Expand_7266', 'Expand_7258', 'Expand_7386', 'Expand_7378',
                  'Expand_7370', 'Expand_7362', 'Expand_8025', 'Expand_8017', 'Expand_8009', 'Expand_8001',
                  'Expand_8121', 'Expand_8113', 'Expand_8105', 'Expand_8129', 'Expand_8233', 'Expand_8225',
                  'Expand_8217', 'Expand_8209', 'Expand_8321', 'Expand_8337', 'Expand_8329', 'Expand_8313',
                  'Expand_8441', 'Expand_8433', 'Expand_8425', 'Expand_8417', 'Expand_8545', 'Expand_8537',
                  'Expand_8529', 'Expand_8521', 'Expand_8649', 'Expand_8633', 'Expand_8641', 'Expand_8625',
                  'Expand_8745', 'Expand_8753', 'Expand_8737', 'Expand_8729', 'Expand_8841', 'Expand_8833',
                  'Expand_8857', 'Expand_8849', 'Expand_9496', 'Expand_9488', 'Expand_9480', 'Expand_9472',
                  'Expand_9600', 'Expand_9592', 'Expand_9584', 'Expand_9576', 'Expand_9704', 'Expand_9696',
                  'Expand_9688', 'Expand_9680', 'Expand_9808', 'Expand_9792', 'Expand_9784', 'Expand_9800',
                  'Expand_9912', 'Expand_9904', 'Expand_9896', 'Expand_9888', 'Expand_10120', 'Expand_10112',
                  'Expand_10104', 'Expand_10096', 'Expand_10016', 'Expand_10008', 'Expand_10000', 'Expand_9992',
                  'Expand_10224', 'Expand_10216', 'Expand_10208', 'Expand_10200', 'Expand_10320', 'Expand_10312',
                  'Expand_10328', 'Expand_10304', 'Expand_10959', 'Expand_10951', 'Expand_10967', 'Expand_10943',
                  'Expand_11071', 'Expand_11063', 'Expand_11047', 'Expand_11055', 'Expand_11175', 'Expand_11159',
                  'Expand_11167', 'Expand_11151', 'Expand_11271', 'Expand_11279', 'Expand_11263', 'Expand_11255',
                  'Expand_11383', 'Expand_11375', 'Expand_11367', 'Expand_11359', 'Expand_11487', 'Expand_11479',
                  'Expand_11471', 'Expand_11463', 'Expand_11591', 'Expand_11583', 'Expand_11575', 'Expand_11567',
                  'Expand_11687', 'Expand_11679', 'Expand_11695', 'Expand_11671', 'Expand_11799', 'Expand_11783',
                  'Expand_11791', 'Expand_11775', 'Expand_12438', 'Expand_12430', 'Expand_12422', 'Expand_12414',
                  'Expand_12542', 'Expand_12534', 'Expand_12518', 'Expand_12526', 'Expand_12646', 'Expand_12638',
                  'Expand_12630', 'Expand_12622', 'Expand_12750', 'Expand_12742', 'Expand_12734', 'Expand_12726',
                  'Expand_12838', 'Expand_12846', 'Expand_12830', 'Expand_12854', 'Expand_12950', 'Expand_12958',
                  'Expand_12942', 'Expand_12934', 'Expand_13062', 'Expand_13054', 'Expand_13046', 'Expand_13038',
                  'Expand_13166', 'Expand_13158', 'Expand_13150', 'Expand_13142', 'Expand_13270', 'Expand_13262',
                  'Expand_13246', 'Expand_13254', 'Expand_13909', 'Expand_13893', 'Expand_13885', 'Expand_13901',
                  'Expand_14013', 'Expand_14005', 'Expand_13989', 'Expand_13997', 'Expand_14117', 'Expand_14109',
                  'Expand_14101', 'Expand_14093', 'Expand_14221', 'Expand_14205', 'Expand_14197', 'Expand_14213',
                  'Expand_14325', 'Expand_14317', 'Expand_14309', 'Expand_14301', 'Expand_14429', 'Expand_14413',
                  'Expand_14405', 'Expand_14421', 'Expand_14533', 'Expand_14525', 'Expand_14517', 'Expand_14509',
                  'Expand_14637', 'Expand_14621', 'Expand_14629', 'Expand_14613', 'Expand_14733', 'Expand_14741',
                  'Expand_14725', 'Expand_14717', 'Expand_15380', 'Expand_15372', 'Expand_15364', 'Expand_15356',
                  'Expand_15484', 'Expand_15476', 'Expand_15468', 'Expand_15460', 'Expand_15588', 'Expand_15580',
                  'Expand_15572', 'Expand_15564', 'Expand_15684', 'Expand_15676', 'Expand_15668', 'Expand_15692',
                  'Expand_15788', 'Expand_15796', 'Expand_15780', 'Expand_15772', 'Expand_15900', 'Expand_15892',
                  'Expand_15884', 'Expand_15876', 'Expand_15988', 'Expand_15980', 'Expand_16004', 'Expand_15996',
                  'Expand_16108', 'Expand_16100', 'Expand_16092', 'Expand_16084', 'Expand_16212', 'Expand_16196',
                  'Expand_16204', 'Expand_16188', 'Expand_16957', 'Expand_16949', 'Expand_16965', 'Expand_16941',
                  'Expand_17061', 'Expand_17053', 'Expand_17045', 'Expand_17069', 'Expand_17165', 'Expand_17157',
                  'Expand_17173', 'Expand_17149', 'Expand_17277', 'Expand_17269', 'Expand_17261', 'Expand_17253',
                  'Expand_17373', 'Expand_17365', 'Expand_17357', 'Expand_17381', 'Expand_17485', 'Expand_17477',
                  'Expand_17469', 'Expand_17461', 'Expand_17581', 'Expand_17589', 'Expand_17573', 'Expand_17565',
                  'Expand_17693', 'Expand_17685', 'Expand_17677', 'Expand_17669', 'Expand_17797', 'Expand_17781',
                  'Expand_17773', 'Expand_17789', 'Expand_19175', 'Expand_19420', 'Expand_18930', 'Expand_18685',
                  'Expand_18440'
                  ]

def add_cast(i, node, mode):
    '''add cast op int64->int32'''
    new_scale_node_0 = onnx.helper.make_node("Cast",
                                             inputs=[node.input[0]],
                                             outputs=['Cast_mod_' + str(i)],
                                             name='Cast_mod_' + str(i),
                                             to=getattr(onnx.TensorProto, mode))
    onnx_model.graph.node.insert(i, new_scale_node_0)
    node.input[0] = 'Cast_mod_' + str(i)
    new_scale_node_1 = onnx.helper.make_node("Cast",
                                             inputs=[node.input[1]],
                                             outputs=['Cast_mod_' + str(i+1)],
                                             name='Cast_mod_' + str(i+1),
                                             to=getattr(onnx.TensorProto, mode))
    onnx_model.graph.node.insert(i, new_scale_node_1)
    node.input[1] = 'Cast_mod_' + str(i+1)

slide = 0

for i, node in enumerate(onnx_model.graph.node):

    if node.name in cast_mod_nodes:
        cast_mod_nodes.remove(node.name)
        if 'Clip' in node.name or 'Div' in node.name:
            print("add cast int64->float32 before", node.name)
            add_cast(i + slide, node, "FLOAT")
        else:
            print("add cast int64->int32 before", node.name)
            add_cast(i+slide, node, "INT32")

onnx.save(onnx_model, '../swin_mod_bs{}.onnx'.format(batchsize))