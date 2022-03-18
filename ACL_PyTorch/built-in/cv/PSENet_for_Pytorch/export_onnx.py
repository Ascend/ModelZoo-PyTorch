import torch
from fpn_resnet_nearest import resnet50
import torch.onnx
from collections import OrderedDict
import torch._utils


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]

        new_state_dict[name] = v
    return new_state_dict


def convert():
    checkpoint = torch.load("./PSENet_nearest.pth", map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
    # model = mobilenet.mobilenet_v2(pretrained = False)
    model = resnet50()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 704, 1216)
    import onnx
    dynamic_axes = {'actual_input_1':{0:'-1'},'output1':{0:'-1'}}
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model, dummy_input, "PSENet_704_1216_nearest.onnx", input_names=input_names, output_names=output_names,dynamic_axes = dynamic_axes, opset_version=11)


if __name__ == "__main__":
    convert()
