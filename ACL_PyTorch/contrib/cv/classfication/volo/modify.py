import numpy as np
from MagicONNX.magiconnx import OnnxGraph
import argparse

INT32_MAX = 2147483647
INT32_MIN = -2147483648

def modify(path, output):
    graph = OnnxGraph(path)
    col2ims = graph.get_nodes("Col2im")
    for idx, node in enumerate(col2ims):
        attr = node['output_size']
        node.attrs.pop("output_size")
        new_init = graph.add_initializer(f'output_size_{node.name}', np.array(attr).astype(np.int32))
        node.inputs = [node.inputs[0], f'output_size_{node.name}']

    graph.save(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='modify the onnx node')
    parser.add_argument('--src', type=str, default='./d1_224_84.2.pth.tar',
                        help='weights of pytorch dir')
    parser.add_argument('--des', type=str, default='./volo_d1_224_Col2im.onnx',
                        help='weights of onnx dir')
    args = parser.parse_args()
    modify(args.src, args.des)
    print("modify the onnx successfully!")


