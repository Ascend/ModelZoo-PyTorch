import torch
import torch.onnx
import mobilenet
from collections import OrderedDict


def proc_nodes_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for key, value in checkpoint[attr_name].items():
        if key == "module.features.0.0.weight":
            print(value)
        if key[0:7] == "module.":
            name = key[7:]
        else:
            name = key[0:]
        
        new_state_dict[name] = value
    return new_state_dict


def convert():
    checkpoint = torch.load("./mobilenet_cpu.pth.tar", map_location=torch.device('cpu'))
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
    model = mobilenet.mobilenet_v2(pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, 
        "mobilenet_v2_npu.onnx",
        input_names=input_names, 
        output_names=output_names, 
        opset_version=11)  # 7


if __name__ == "__main__":
    convert()
