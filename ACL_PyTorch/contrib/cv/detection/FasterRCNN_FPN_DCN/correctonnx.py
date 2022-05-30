import numpy as np
import onnx
import argparse

# 1:Get batch parameter
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()
print(args.batch_size)

onnx_model = onnx.load("./faster_rcnn_r50_fpn_1x_coco.onnx")
print("Load successful")
graph = onnx_model.graph
node = graph.node

cont = 0
node_constant_list = []
for i in range(len(node)):
    if node[i].op_type == 'Constant':
        for attr_id, attr in enumerate(node[i].attribute):
            if attr.t.data_type == 6:
                # 2:Old nodes ready to be deleted
                old_scale_node = node[i]
                data = np.ones((args.batch_size, attr.t.dims[1], attr.t.dims[2], attr.t.dims[3])).astype(np.float32)
                # 3:Prepare new node
                attr.t.dims[0] = args.batch_size
                new_scale_node = onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=node[i].output,
                    # value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, dims=[args.batch_size, attr.t.dims[1], attr.t.dims[2], attr.t.dims[3]],vals=data.tobytes(), raw=True)

                    value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT,
                                                  dims=[attr.t.dims[0], attr.t.dims[1], attr.t.dims[2], attr.t.dims[3]],
                                                  vals=data.tobytes(), raw=True)
                )  # 4:Create a new node

                # print(new_scale_node)

                graph.node.remove(old_scale_node)  # Delete old node
                graph.node.insert(i, new_scale_node)  # Insert new node

                cont = cont + 1
                print("Complete once node modification")

str_bs = str(args.batch_size)
new_model_name = './faster_rcnn_r50_fpn_1x_coco_change_bs' + str_bs + '.onnx'
onnx.save(onnx_model, new_model_name)
print(cont)

