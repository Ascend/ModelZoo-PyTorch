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

BATCH_SIZE = 1
Pth_Model_Path = './mbv3_small.pth.tar'
model_pt = MobileNetV3_Small()

checkpoint = torch.load(Pth_Model_Path, map_location='cpu')['state_dict']
checkpoint = adjust_checkpoint(checkpoint)

model_pt.load_state_dict(checkpoint)
model_pt.eval()

inputs = torch.randn(BATCH_SIZE, 3, 224, 224)
print("Start compiling MobileNetV3 model...")

ts_model = torch.jit.trace(model_pt, inputs)
ts_path = "./mobilenetv3.ts"
ts_model.save(ts_path)
print(f"torchscript has been successfully exported to {ts_path}")
