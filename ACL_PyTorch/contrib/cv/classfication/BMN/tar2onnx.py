import torch
import os

# 1. Create model
from models import BMN
import opts

opt = opts.parse_opt()
opt = vars(opt)

input_file = opt['input_file']
output_file = opt['output_file']
batch_size = opt['infer_batch_size']
opset_version = opt['opset_version']
verbose = opt['verbose']

feat_dim = opt['feat_dim']
temporal_scale = opt['temporal_scale']

print('Model checkpoint path is:', os.path.abspath(opt['input_file']))
model = BMN(opt)
checkpoint = torch.load(input_file, map_location=torch.device('cpu'))['state_dict']

# 2. rm 'module.', load state_dict
state_dict = dict()
for k in checkpoint:
    assert k.startswith('module.')
    state_dict[k[7:]] = checkpoint[k]

model.load_state_dict(state_dict)
model.eval()

# 3. Export as onnx
assert feat_dim, temporal_scale == [400, 100]
dummy_input = torch.randn(batch_size, feat_dim, temporal_scale)

import time
start_time = time.time()

torch.onnx.export(model, dummy_input, output_file, input_names=['image'], output_names=['confidence_map', 'start', 'end'], opset_version=opset_version, verbose=verbose)
print('Onnx convertion takes time: {}(s)'.format(time.time()-start_time))
