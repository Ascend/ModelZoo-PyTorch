import numpy as np
from auto_optimizer import OnnxGraph

g = OnnxGraph.parse('pse.onnx')
resize_list = g.get_nodes('Resize')
for node in resize_list:
	node['coordinate_transformation_mode'] = 'pytorch_half_pixel'
	node['cubic_coeff_a'] = -0.75
	node['mode'] = 'linear'
	node['nearest_mode'] = 'floor'
	g[node.inputs[1]].value = np.array([], dtype=np.float32)
g.save('pse_new.onnx')
