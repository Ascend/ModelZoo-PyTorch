import sys
import numpy as np
from auto_optimizer import OnnxGraph


def modify_onnx(input_onnx, output_onnx):
	g = OnnxGraph.parse(input_onnx)
	g.inputs[0].shape = ["-1", 3, 736, 1280]
	g.infershape()

	conv2_bias = g.add_initializer(
		'conv2d_transpose_0.b', g['conv2d_transpose_0.b_0'].value.flatten())
	g['p2o.ConvTranspose.0'].inputs.append('conv2d_transpose_0.b')

	conv3_bias = g.add_initializer(
		'conv2d_transpose_1.b', g['conv2d_transpose_1.b_0'].value.flatten())
	g['p2o.ConvTranspose.2'].inputs.append('conv2d_transpose_1.b')

	g.remove('p2o.Add.98')
	g.remove('p2o.Add.100')
	g.remove('conv2d_transpose_0.b_0')
	g.remove('conv2d_transpose_1.b_0')

	g.update_map()
	g.save(output_onnx)


if __name__ == '__main__':
	modify_onnx(sys.argv[1], sys.argv[2])
