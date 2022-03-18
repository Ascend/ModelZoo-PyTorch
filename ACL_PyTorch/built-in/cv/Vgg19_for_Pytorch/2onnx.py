import torch
import torch.onnx
from collections import OrderedDict
import ssl
import torchvision.models as models

def convert():
    #checkpoint = torch.load("./efficientnet_74.9_100epoch_checkpoint.pth", map_location='cpu')
    #checkpoint['state_dict'] = proc_nodes_module(checkpoint,'state_dict')
    model = models.vgg19(pretrained=True)
    #model.set_swish(memory_efficient=False)
    #model.load_state_dict(checkpoint['state_dict'],strict=False)
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    
    torch.onnx.export(model, dummy_input, "vgg19.onnx", input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11)

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    convert()
