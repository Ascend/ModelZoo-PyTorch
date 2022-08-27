import sys
import onnx
from onnx import helper, optimizer

input_model=sys.argv[1]
output_model=sys.argv[2]
model = onnx.load(input_model)
# onnx.checker.check_model(model)
model_nodes = model.graph.node


def getNodeByName(nodes, name: str):
    for n in nodes:
        if n.name == name:
            return n
    
    return -1

# fix shape for resize
sizes1 = onnx.helper.make_tensor('size1', onnx.TensorProto.INT32, [4], [1, 256, 76, 76])
sizes2 = onnx.helper.make_tensor('size2', onnx.TensorProto.INT32, [4], [1, 256, 152, 152])
sizes3 = onnx.helper.make_tensor('size3', onnx.TensorProto.INT32, [4], [1, 256, 304, 304])
model.graph.initializer.append(sizes1)
model.graph.initializer.append(sizes2)
model.graph.initializer.append(sizes3)

getNodeByName(model_nodes, 'Resize_141').input[3] = "size1"
getNodeByName(model_nodes, 'Resize_161').input[3] = "size2"
getNodeByName(model_nodes, 'Resize_181').input[3] = "size3"


print("Faster R-CNN onnx adapted to ATC")
onnx.save(model, output_model)