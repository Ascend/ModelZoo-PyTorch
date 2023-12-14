from collections import OrderedDict
import torch
from mobilenetv3 import MobileNetV3_Small

def adjust_checkpoint(checkpoint):
    new_state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if key == "module.features.0.0.weight":
            print(value)
        if key[0:7] == "module.":
            name = key[7:]
        else:
            name = key[0:]
        
        new_state_dict[name] = value
    return new_state_dict

def torchscript_export(batch_size):
    Pth_Model_Path = './mbv3_small.pth.tar'
    model_pt = MobileNetV3_Small()

    checkpoint = torch.load(Pth_Model_Path, map_location='cpu')['state_dict']
    checkpoint = adjust_checkpoint(checkpoint)

    model_pt.load_state_dict(checkpoint)
    model_pt.eval()

    inputs = torch.randn(batch_size, 3, 224, 224)
    print("Start exporting MobileNetV3 torchscript model...")

    ts_model = torch.jit.trace(model_pt, inputs)
    ts_path = f"./mobilenetv3_bs{batch_size}.ts"
    ts_model.save(ts_path)
    print(f"torchscript has been successfully exported to {ts_path}")
    return ts_path