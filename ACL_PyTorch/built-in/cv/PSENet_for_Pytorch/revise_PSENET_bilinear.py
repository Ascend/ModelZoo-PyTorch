import numpy as np
import onnx
import onnxruntime as rt
from onnx import shape_inference
import sys
model_path = sys.argv[1]
model = onnx.load(model_path)

# model = onnx.shape_inference.infer_shapes(model)

def getNodeAndIOname(nodename,model):
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == nodename:
            Node = model.graph.node[i]
    return Node,input_name,output_name

def FindPeerOutNode(graph, edge_name):
    for i, x in enumerate(graph.node):
        if edge_name in x.output:
            return i
    return -1


def RemoveNode(graph, node_list):
    cnt = 0
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name in node_list:
            graph.node.remove(graph.node[i - cnt])  # 因为节点个数变少了
            cnt += 1
def FindDependNode(graph, end_node, start_node):
    '''
    find dependency node, [end_node, start_node)
    '''
    def dfs(graph, idx, start_node, n_list):
        for edge in graph.node[idx].input:
            node_idx = FindPeerOutNode(graph, edge)
            if node_idx < 0:
                # print('bad peerout index')
                continue
            n = graph.node[node_idx]
            if n.name != start_node:
                n_list.append(n.name)
                # print('n.name', n.name)
                n_list = dfs(graph, node_idx, start_node, n_list)
        return n_list

    index = GetNodeIndex(graph, end_node)
    n_list = [end_node, ]
    return dfs(graph, index, start_node, n_list)


def createGraphMemberMap(graph_member_list):
    member_map=dict();
    for n in graph_member_list:
        member_map[n.name]=n;
    return member_map

def GetNodeIndex(graph, node_name):
    index = 0
    for i in range(len(graph.node)):
        if graph.node[i].name == node_name:
            index = i
            break
    return index
def RemoveNode2(graph,node_list):
    for name in node_list:
        print("name",name)
        ind = GetNodeIndex(graph,name)
        print("ind:",ind)
        graph.node.remove(graph.node[ind])

    
for i in range(len(model.graph.node)):
    if model.graph.node[i].op_type == "Resize":
        print("Resize", i, model.graph.node[i].input, model.graph.node[i].output)

sizes1 = onnx.helper.make_tensor('size1', onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
sizes2 = onnx.helper.make_tensor('size2', onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
sizes3 = onnx.helper.make_tensor('size3', onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
sizes4 = onnx.helper.make_tensor('size4', onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
sizes5 = onnx.helper.make_tensor('size5', onnx.TensorProto.FLOAT, [4], [1, 1, 4, 4])
sizes6 = onnx.helper.make_tensor('size6', onnx.TensorProto.FLOAT, [4], [1, 1, 8, 8])
sizes7 = onnx.helper.make_tensor('size7', onnx.TensorProto.FLOAT, [4], [1, 1, 4, 4])


model.graph.initializer.append(sizes1)
model.graph.initializer.append(sizes2)
model.graph.initializer.append(sizes3)
model.graph.initializer.append(sizes4)
model.graph.initializer.append(sizes5)
model.graph.initializer.append(sizes6)
model.graph.initializer.append(sizes7)


newnode = onnx.helper.make_node(
    'Resize',
    name='Resize_196',
    # inputs=['551', '564', '572', 'size1'],
    inputs=['551', '564', 'size1'],
    outputs=['573'],
    coordinate_transformation_mode='pytorch_half_pixel',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)

newnode2 = onnx.helper.make_node(
    'Resize',
    name='Resize_224',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['579', '592', 'size2'],
    outputs=['601'],
    coordinate_transformation_mode='pytorch_half_pixel',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)

newnode3 = onnx.helper.make_node(
    'Resize',
    name='Resize_252',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['607', '620', 'size3'],
    outputs=['629'],
    coordinate_transformation_mode='pytorch_half_pixel',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)

newnode4 = onnx.helper.make_node(
    'Resize',
    name='Resize_285',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['607', '653', 'size4'],
    outputs=['662'],
    coordinate_transformation_mode='pytorch_half_pixel',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)


newnode5 = onnx.helper.make_node(
    'Resize',
    name='Resize_312',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['579', '680', 'size5'],
    outputs=['689'],
    coordinate_transformation_mode='pytorch_half_pixel',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)


newnode6 = onnx.helper.make_node(
    'Resize',
    name='Resize_339',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['551', '707', 'size6'],
    outputs=['716'],
    coordinate_transformation_mode='pytorch_half_pixel',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)


newnode7= onnx.helper.make_node(
    'Resize',
    name='Resize_371',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['721', '739', 'size7'],
    outputs=['output1'],
    coordinate_transformation_mode='pytorch_half_pixel',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)




model.graph.node.remove(model.graph.node[196])
model.graph.node.insert(196, newnode)

model.graph.node.remove(model.graph.node[224])
model.graph.node.insert(224, newnode2)

model.graph.node.remove(model.graph.node[252])
model.graph.node.insert(252, newnode3)

model.graph.node.remove(model.graph.node[285])
model.graph.node.insert(285, newnode4)

model.graph.node.remove(model.graph.node[312])
model.graph.node.insert(312, newnode5)

model.graph.node.remove(model.graph.node[339])
model.graph.node.insert(339, newnode6)

model.graph.node.remove(model.graph.node[371])
model.graph.node.insert(371, newnode7)

slice_node1_1 = FindDependNode(model.graph, 'Slice_192', 'Relu_174') #结尾（will be deleted） qishi
print('node map:', slice_node1_1)

slice_node1_2 = FindDependNode(model.graph, 'Cast_193', 'Relu_177')
print('node map:', slice_node1_2)

slice_node2_1 = FindDependNode(model.graph, 'Slice_220', 'Relu_202')
print('node map:', slice_node2_1)

slice_node2_2 = FindDependNode(model.graph, 'Cast_221', 'Relu_205')
print('node map:', slice_node2_2)

slice_node3_1 = FindDependNode(model.graph, 'Slice_248', 'Relu_230')
print('node map:', slice_node3_1)
slice_node3_2 = FindDependNode(model.graph, 'Cast_249', 'Relu_233')
print('node map:', slice_node3_2)


slice_node4_1 = FindDependNode(model.graph, 'Slice_281', 'Relu_230')
print('node map:', slice_node4_1)
slice_node4_2 = FindDependNode(model.graph, 'Cast_282', 'Relu_258')
print('node map:', slice_node4_2)


slice_node5_1 = FindDependNode(model.graph, 'Slice_308', 'Relu_202')
print('node map:', slice_node5_1)
slice_node5_2 = FindDependNode(model.graph, 'Cast_309', 'Relu_258')
print('node map:', slice_node5_2)

slice_node6_1 = FindDependNode(model.graph, 'Slice_335', 'Relu_174')
print('node map:', slice_node6_1)
slice_node6_2 = FindDependNode(model.graph, 'Cast_336', 'Relu_258')
print('node map:', slice_node6_2)

slice_node7_1 = FindDependNode(model.graph, 'Slice_367', 'Conv_344')
print('node map:', slice_node7_1)
slice_node7_2 = FindDependNode(model.graph, 'Cast_368', 'actual_input_1')
print('node map:', slice_node7_2)


node_list = []
node_list.extend(slice_node1_1)
node_list.extend(slice_node1_2)
node_list.extend(slice_node2_1)
node_list.extend(slice_node2_2)
node_list.extend(slice_node3_1)
node_list.extend(slice_node3_2)
node_list.extend(slice_node4_1)
node_list.extend(slice_node4_2)
node_list.extend(slice_node5_1)
node_list.extend(slice_node5_2)
node_list.extend(slice_node6_1)
node_list.extend(slice_node6_2)
node_list.extend(slice_node7_1)
node_list.extend(slice_node7_2)
node_list.extend(['Concat_194'])

node_list.extend(['Concat_222'])

node_list.extend(['Concat_250'])

node_list.extend(['Concat_283'])

node_list.extend(['Concat_337'])

node_list.extend(['Concat_369'])
node_list.extend(['Concat_310'])
#node_list.extend(['Concat_308','Constant_140','Constant_166','Constant_192','Constant_224','Constant_251','Constant_278','Constant_301','Constant_309'])
print(node_list)
RemoveNode2(model.graph, node_list)

#移除最后一个Resize
# 去除最后一个resize节点
node_list=[]
node_list.extend(['Resize_371'])
print(node_list)
RemoveNode2(model.graph, node_list)  #将最后一个Resize节点移除
#将ouput1移除，并建立一个新的，插入进去

out0_info = onnx.helper.make_tensor_value_info('721', onnx.TensorProto.FLOAT, [-1, 7, 176, 304])
model.graph.output.remove(model.graph.output[0])
model.graph.output.insert(0, out0_info)

onnx.checker.check_model(model)


onnx.save(model, sys.argv[1].split('.')[0] + "_revised.onnx")
# m = onnx.load("modify.onnx")



