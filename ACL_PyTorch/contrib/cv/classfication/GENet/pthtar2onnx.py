import torch
import torch.onnx
from collections import OrderedDict
import sys
sys.path.append('./pytorch-GENet')
from models import *


def proc_node_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def convert(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')['state_dict']
    checkpoint = proc_node_module(checkpoint)
    model = WideResNet(16, 8, num_classes=10, mlp=False, extra_params=False)
    model.load_state_dict(checkpoint, strict=False)  
    model.eval()
    print(model)
    
    input_names = ['image']
    output_names = ['class']
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy_input, "genet_gpu.onnx"
                     , input_names=input_names, output_names=output_names, dynamic_axes = dynamic_axes
                     , opset_version=11, verbose=True)


if __name__ == "__main__":
    model_path = sys.argv[1]
    convert(model_path)
