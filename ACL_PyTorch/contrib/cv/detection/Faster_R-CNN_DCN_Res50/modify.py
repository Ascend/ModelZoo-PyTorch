from auto_optimizer import OnnxGraph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="./faster_rcnn_r50_fpn.onnx")
parser.add_argument('--output', type=str, default="./faster_rcnn_r50_fpn_m.onnx")

if __name__ == '__main__':
	opts = parser.parse_args()
	model=OnnxGraph.parse(opts.model)
	model.remove('Split_434')
	model.save(opts.output)