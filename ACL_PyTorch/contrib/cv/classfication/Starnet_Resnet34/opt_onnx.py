import argparse
from auto_optimizer import OnnxGraph
def create_grid_sample(onnx_in, onnx_out):
    graph = OnnxGraph.parse(onnx_in)
    graph.remove('Unsqueeze_8', maps={})
    graph.remove('Pad_0', maps={})
    graph.remove('Squeeze_9', maps={})
    graph.remove('Add_9', maps={})
    graph.remove('Transpose_6', maps={})
    gridsample = graph.add_node('D_gridsample', 'GridSample', [], [], {'padding_mode':b'zeros', 'mode':b'bilinear','align_corners':1})
    graph['D_gridsample'].inputs=['transpose_3.tmp_0', 'transpose_4.tmp_0']
    graph['D_gridsample'].outputs=['tmp_11']
    graph.update_map()
    graph.save(onnx_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="opt onnx")  # task process paramater
    parser.add_argument('--in_onnx', type=str)
    parser.add_argument('--out_onnx', type=str)
    args = parser.parse_args()

    create_grid_sample(args.in_onnx, args.out_onnx)
    
    print("[info] Optimize onnx success. result onnx is: {}".format(args.out_onnx))