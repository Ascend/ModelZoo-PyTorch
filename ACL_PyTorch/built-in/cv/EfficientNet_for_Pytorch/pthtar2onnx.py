import torch
#from efficientnet_pytorch import EfficientNet
from NPU.efficientnet_pytorch import EfficientNet
import torch.onnx

from collections import OrderedDict

def proc_nodes_module(checkpoint,AttrName):
    new_state_dict = OrderedDict()
    for k,v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        
        new_state_dict[name]=v
    return new_state_dict

def convert():
    checkpoint = torch.load("./checkpoint.pth.140.ok.cpu", map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint,'state_dict')
    model = EfficientNet.from_name('efficientnet-b0')
    model.set_swish(memory_efficient=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    #print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 224, 224)
    #dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    torch.onnx.export(model, dummy_input, "efficientnet_tr.onnx", input_names = input_names, output_names = output_names, opset_version=11)
    #torch.onnx.export(model, dummy_input, "efficientnet_dynamic.onnx", input_names = input_names, output_names = output_names, dynamic_axes = dynamic_axes, opset_version=11)

if __name__ == "__main__":
    convert()
