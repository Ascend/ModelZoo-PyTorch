import torch
from baseline.model import DeepMAR
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
    checkpoint = torch.load("./checkpoint.pth.tar", map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
    model = DeepMAR.DeepMAR_ResNet50()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 3, 224, 224)
    import onnx
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model, dummy_input, "Deepmar_bs1.onnx", input_names=input_names, output_names=output_names,
                      opset_version=11, do_constant_folding=True)


if __name__ == "__main__":
    convert()
