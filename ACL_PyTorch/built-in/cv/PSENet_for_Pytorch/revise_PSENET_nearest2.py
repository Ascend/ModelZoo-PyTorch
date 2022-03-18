import numpy as np
import onnx
import onnxruntime as rt
from onnx import shape_inference
import sys


model_path = sys.argv[1]
model = onnx.load(model_path)


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
    name='Resize_141',
    # inputs=['551', '564', '572', 'size1'],
    inputs=['551', '564', 'size1'],
    outputs=['573'],
    coordinate_transformation_mode='asymmetric',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)

newnode2 = onnx.helper.make_node(
    'Resize',
    name='Resize_165',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['577', '590', 'size2'],
    outputs=['599'],
    coordinate_transformation_mode='asymmetric',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)

newnode3 = onnx.helper.make_node(
    'Resize',
    name='Resize_189',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['603', '616', 'size3'],
    outputs=['625'],
    coordinate_transformation_mode='asymmetric',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)

newnode4 = onnx.helper.make_node(
    'Resize',
    name='Resize_219',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['603', '647', 'size4'],
    outputs=['656'],
    coordinate_transformation_mode='asymmetric',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)


newnode5 = onnx.helper.make_node(
    'Resize',
    name='Resize_246',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['577', '674', 'size5'],
    outputs=['683'],
    coordinate_transformation_mode='asymmetric',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)


newnode6 = onnx.helper.make_node(
    'Resize',
    name='Resize_273',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['551', '701', 'size6'],
    outputs=['710'],
    coordinate_transformation_mode='asymmetric',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)


newnode7= onnx.helper.make_node(
    'Resize',
    name='Resize_304',
    # inputs=['347', '367', '375', 'size2'],
    inputs=['715', '733', 'size7'],
    outputs=['output1'],
    coordinate_transformation_mode='asymmetric',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor'
)


model.graph.node.remove(model.graph.node[141])
model.graph.node.insert(141, newnode)

model.graph.node.remove(model.graph.node[165])
model.graph.node.insert(165, newnode2)

model.graph.node.remove(model.graph.node[189])
model.graph.node.insert(189, newnode3)

model.graph.node.remove(model.graph.node[219])
model.graph.node.insert(219, newnode4)

model.graph.node.remove(model.graph.node[246])
model.graph.node.insert(246, newnode5)

model.graph.node.remove(model.graph.node[273])
model.graph.node.insert(273, newnode6)

model.graph.node.remove(model.graph.node[304])
model.graph.node.insert(304, newnode7)

slice_node1_1 = FindDependNode(model.graph, 'Slice_137', 'Relu_120') #结尾（will be deleted） qishi
print('node map:', slice_node1_1)

slice_node1_2 = FindDependNode(model.graph, 'Cast_138', 'Relu_122')
print('node map:', slice_node1_2)

slice_node2_1 = FindDependNode(model.graph, 'Slice_161', 'Relu_144')
print('node map:', slice_node2_1)

slice_node2_2 = FindDependNode(model.graph, 'Cast_162', 'Relu_146')
print('node map:', slice_node2_2)

slice_node3_1 = FindDependNode(model.graph, 'Slice_185', 'Relu_168')
print('node map:', slice_node3_1)
slice_node3_2 = FindDependNode(model.graph, 'Cast_186', 'Relu_170')
print('node map:', slice_node3_2)


slice_node4_1 = FindDependNode(model.graph, 'Slice_215', 'Relu_168')
print('node map:', slice_node4_1)
slice_node4_2 = FindDependNode(model.graph, 'Cast_216', 'Relu_192')
print('node map:', slice_node4_2)


slice_node5_1 = FindDependNode(model.graph, 'Slice_242', 'Relu_144')
print('node map:', slice_node5_1)
slice_node5_2 = FindDependNode(model.graph, 'Cast_243', 'Relu_192')
print('node map:', slice_node5_2)

slice_node6_1 = FindDependNode(model.graph, 'Slice_269', 'Relu_120')
print('node map:', slice_node6_1)
slice_node6_2 = FindDependNode(model.graph, 'Cast_270', 'Relu_192')
print('node map:', slice_node6_2)

slice_node7_1 = FindDependNode(model.graph, 'Slice_300', 'Conv_277')
print('node map:', slice_node7_1)
slice_node7_2 = FindDependNode(model.graph, 'Cast_301', 'actual_input_1')
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
node_list.extend(['Concat_139'])
node_list.extend(['Concat_163'])
node_list.extend(['Concat_187'])
node_list.extend(['Concat_217'])
node_list.extend(['Concat_271'])
node_list.extend(['Concat_302'])
node_list.extend(['Concat_244'])
print(node_list)
RemoveNode2(model.graph, node_list)


onnx.checker.check_model(model)
onnx.save(model, sys.argv[1].split('.')[0] + "_revised2.onnx")
