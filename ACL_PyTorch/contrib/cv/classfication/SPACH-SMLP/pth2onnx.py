
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from timm.models import create_model
from easydict import EasyDict
import torch
# 需要导入，以register_model
import models

def parse_model_args(args):
    model = args.model
    model_args = []
    if model.startswith('spach'):
        model_args = ['stem_type', 'shared_spatial_func']
    args = vars(args)
    model_args = {_: args[_] for _ in model_args}
    return model_args

def main():
    args = EasyDict(aa='rand-m9-mstd0.5-inc1', auto_resume=True, batch_size=128, clip_grad=None, color_jitter=0.4, cooldown_epochs=10, cutmix=1.0, cutmix_minmax=None, data_path='/opt/gpu/imagenet/', data_set='IMNET', decay_epochs=30, decay_rate=0.1, device='cuda', dist_eval=True, dist_url='env://', distillation_alpha=0.5, distillation_tau=1.0, distillation_type='none', distributed=False, drop=0.0, drop_path=0.1, epochs=300, eval=False, finetune='', inat_category='name', input_size=224, lr=0.0005, lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, min_lr=1e-05, mixup=0.8, mixup_mode='batch', mixup_prob=1.0, mixup_switch_prob=0.5, model='smlpnet_tiny', model_ema=True, model_ema_decay=0.99996, model_ema_force_cpu=False, momentum=0.9, nb_classes=1000, num_workers=10, opt='adamw', opt_betas=None, opt_eps=1e-08, output_dir='./output2/', patience_epochs=10, pin_mem=True, recount=1, remode='pixel', repeated_aug=True, reprob=0.25, resplit=False, resume='', sched='cosine', seed=0, shared_spatial_func=False, smoothing=0.1, start_epoch=0, stem_type='conv1', teacher_model='regnety_160', teacher_path='', throughput=False, train_interpolation='bicubic', warmup_epochs=20, warmup_lr=1e-06, weight_decay=0.05, world_size=1)
    # args.model = 'smlpnet_tiny'
    # args.nb_classes = 1000
    # args.drop = 0.0
    # args.drop_path = 0.1

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        **parse_model_args(args)
    )
    checkpoint_model_name = "smlp_t.pth"
    checkpoint = torch.load(checkpoint_model_name, map_location='cpu')
    # print(checkpoint.keys()) # dict_keys(['model', 'optimizer', 'lr_scheduler', 'epoch', 'model_ema', 'scaler', 'args'])
    model.load_state_dict(checkpoint["model"], strict=False)
    

    # 模型设置为推理模式
    model.eval()

    batch_size = 1  #批处理大小
    input_shape = (3, 224, 224)   #输入数据,改成自己的输入shape
    dummy_input = torch.randn(batch_size, *input_shape) #  定义输入shape

    torch.onnx.export(model, 
                        dummy_input, 
                        "sMLPNet-T.onnx", 
                        input_names = ["input"],   # 构造输入名
                        output_names = ["output"],    # 构造输出名
                        opset_version=11,    # ATC工具目前仅支持opset_version=11
                        dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}})  #支持输出动态轴
                        
main()
