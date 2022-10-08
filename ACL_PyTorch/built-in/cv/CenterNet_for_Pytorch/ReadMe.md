# 参考论文

- [Objects as Points](https://arxiv.org/abs/1904.07850)

# 参考实现

- [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)
- [CaoWGG/TensorRT-CenterNet](https://github.com/CaoWGG/TensorRT-CenterNet)

# 环境搭建

1、创建一个conda环境

```shell
conda create --nameCenterNet python=3.6
```

激活环境

```
conda activate CenterNet
```

2、clone仓库

```
git clone https://github.com/xingyizhou/CenterNet
```

3、安装依赖

```
cd CenterNet
pip install -r requirements.txt
```

4、安装pytorch v1.0.0

1) 下载[torch-1.0.0-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl)

2) 安装

```
pip install torch-1.0.0-cp36-cp36m-linux_x86_64.whl
```

**注意：**1) pytorch版本必须是1.0.0；2) 需确保是GPU环境；3) CUDA版本为10.2

5、安装其它依赖

```
pip install tqdm==4.19.9 torchvision==0.2.2 onnx==1.8.1 onnxruntime==1.7.0 skl2onnx==1.8.0
```

6、确保gcc和g++版本>=7.2.0

7、安装COCO API

```
cd CenterNet
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
python setup.py install --user
```

8、把THC-based DCNv2替换为ATen-based DCNv2

```
cd CenterNet
git clone https://github.com/CaoWGG/TensorRT-CenterNet.gitcp -r TensorRT-CenterNet/readme/dcn src/lib/models/networks
```

**说明：**主要用的是TensorRT-CenterNet仓下的dcn目录，也可以仅下载dcn目录，然后放入到`CenterNet/src/lib/models/networks`目录下。

9、编译Deform Conv

```
cd src/lib/models/networks/dcn
python setup.py build_ext --inplace
```

**注意：**gcc和g++版本必须>=7.2.0，否则可能导致出错。

10、Change import

把`CenterNet/src/lib/models/networks/pose_dla_dcn.py`和`CenterNet/src/lib/models/networks/resnet_dcn.py`中的`from .DCNv2.dcn_v2 import DCN`改为`from .dcn.modules.deform_conv import ModulatedDeformConvPack as DCN`

11、打开`/root/anaconda3/envs/CenterNet/lib/python3.6/site-packages/torch/autograd/function.py`，定位到273行，把`_iter_filter(...)`函数改为如下：

```
def _iter_filter(condition, allow_unknown=False, condition_msg=None,
                 conversion=None):
    def _iter(obj):
        if conversion is not None:
            obj = conversion(obj)
        if condition(obj):
            yield obj
        #M<<<<<<
        elif isinstance(obj,int):  ## int to tensor
            yield torch.tensor(obj)
        #>>>>>>
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                for var in _iter(o):
                    yield var
        elif allow_unknown:
            yield obj
        else:
            raise ValueError("Auto nesting doesn't know how to process "
                             "an input object of type " + torch.typename(obj) +
                             (". Accepted types: " + condition_msg +
                              ", or lists/tuples of them"
                              if condition_msg else ""))


    return _iter
```

12、下载[ctdet_coco_dla_2x.pth](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT)模型，放入`CenterNet/models`目录下
13、把`CenterNet/src/lib/opts.py中的add_argument('task', default='ctdet'....)`改为`add_argument('--task', default='ctdet'....)`
14、把提供的代码和脚本放入`CenterNet/src`目录下。

# 准备数据集

根据CenterNet官方数据集安装指导准备数据集：[DATA.md](https://github.com/xingyizhou/CenterNet/blob/master/readme/DATA.md)，本示例以 **COCO 2017 Val** 数据集为例。

# PyTorch在线推理

由于后续导出onnx时需要修改CenterNet源码，修改后的代码无法进行PyTorch在线推理。因此这里先进行PyTorch在线推理验证。

运行pth_eval.py进行推理，推理完毕之后会输入精度和推理时间信息。

```
python pth_eval.py --res_data_save_path=./pth_result
```

参数说明：
  - --res_data_save_path：推理结果保存路径

# om模型推理模型转换

1. 模型转换。

   使用PyTorch将模型权重文件pth转换为onnx文件，再使用atc工具将onnx文件转为离线推理模型om文件。

   - 导出onnx文件。

      - 打开`CenterNet/src/lib/models/networks/dcn/functions/deform_conv.py`文件

         - 修改`ModulatedDeformConvFunction`的`symbolic(...)`函数，把原函数改为如下：

            ```
            @staticmethod
            def symbolic(g, input, weight, offset, bias, stride, padding, dilation, groups, deformable_groups):
          return g.op("DeformableConv2D",
                    input,
              weight,
                    offset,
                    bias,
                    deformable_groups_i=deformable_groups,
              dilations_i=dilation,
                    groups_i=groups,
              pads_i=padding,        strides_i=stride)
            ```
      
         - 修改`ModulatedDeformConvFunction`的`forward(...)`函数，把原函数改为如下：

            ```
      @staticmethod
            def forward(ctx, input, weight, offset, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
                ctx.stride = stride
                ctx.padding = padding
          ctx.dilation = dilation
                ctx.groups = groups
          ctx.deformable_groups = deformable_groups
                ctx.with_bias = bias is not None
                if not ctx.with_bias:
                    bias = input.new_empty(1)  # fake tensor
          output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
                return output
            ```
      - 打开`CenterNet/src/lib/models/networks/dcn/modules/deform_conv.py`文件，修改`ModulatedDeformConvPack`的`forward(...)`函数，把原函数改为如下：

           ```
           def forward(self, x):
               out = self.conv_offset_mask(x)
               o1, o2, mask = torch.chunk(out, 3, dim=1)
               offset = torch.cat((o1, o2), dim=1)
               mask = torch.sigmoid(mask)

               offset_y = offset.reshape(1, -1, 2, offset.shape[2],
                     offset.shape[3])[:, :, 0, ...].reshape(1, offset.shape[1] // 2, offset.shape[2],
                        offset.shape[3])
               offset_x = offset.reshape(1, -1, 2, offset.shape[2],
                     offset.shape[3])[:, :, 1, ...].reshape(1, offset.shape[1] // 2, offset.shape[2],
                        offset.shape[3])
               offset = torch.cat((offset_x, offset_y, mask), 1)

               return modulated_deform_conv(x, self.weight, offset, self.bias, self.stride, self.padding, self.dilation,         self.groups, self.deformable_groups)
         ```
      - 打开`/root/anaconda3/envs/CenterNet/lib/python3.6/site-packages/torch/onnx/symbolic.py`，在`reciprocal(...)`函数后边增加两个函数：
      
         ```
         def reshape(g, self, shape):
          return view(g, self, shape)
         
         
         def reshape_as(g, self, other):
          shape = g.op('Shape', other)
             return reshape(g, self, shape)
         ```
      
      - 运行`export_onnx.py`文件，导出onnx模型
      
         ```
         python export_onnx.py
         ```
      
         运行完之后，会在`CenterNet/models`目录下生成`ctdet_coco_dla_2x.onnx`模型文件。
      
      - 运行`modify_onnx.py`文件，修改onnx模型文件
      
         ```
         python modify_onnx.py
         ```
     
       运行完之后，会在`CenterNet/models`目录下生成`ctdet_coco_dla_2x_modify.onnx`模型文件。

   2. 使用atc工具将onnx模型转om模型。
   
      - 根据实际情况，修改`onnx2om.sh`脚本中的环境变量，具体的脚本示例如下：
        
         ```
         #!/bin/bash
         clear
         
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         
         atc --framework=5 --model=../models/ctdet_coco_dla_2x_modify.onnx --output=../models/ctdet_coco_dla_2x \
         --input_format=NCHW --input_shape=image:1,3,512,512 --log=error --soc_version=Ascend310
         rm -rf fusion_result.json kernel_meta
         ```
         
         参数说明：
         
          - --model：为onnx模型文件
          - --framework：5代表onnx模型
         
         - --output：输出的om模型
         - --input_format：输入数据的格式
         - --input_shape：输入数据的shape
         - --log：日志等级
         - --soc_version：部署芯片类型
         
      - 执行onnx2om.sh脚本，将onnx文件转为离线推理模型文件om文件。
      
         ```
          bash onnx2om.sh
         ```
         
          运行完之后，会在`CenterNet/models`目录下生成`ctdet_coco_dla_2x.om`模型文件。
   
2. 开始推理验证。

   - 根据实际情况，修改`benchmark_infer.sh`脚本中的代码，具体的脚本示例如下：

      ```
      #!/bin/bash
      clear
      
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      
      ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=../models/ctdet_coco_dla_2x.om \-input_text_path=./pre_bin/bin_file.info -input_width=512 -input_height=512 -output_binary=true -useDvpp=false
      ```
      
   - 执行`benchmark_infer.sh`脚本，进行benchmark推理。

      ```
      sh benchmark_infer.sh
      ```

   - 推理数据后处理与精度统计。
      运行postprocess.py脚本，进行后处理和精度统计。

      ```
      python postprocess.py \
        --infer_res_save_path=./result/dumpOutput_device0 \
        --pre_data_save_path=./pre_bin \
        --res_data_save_path=./om_result
      ```

      参数说明：

      - --infer_res_save_path：benchmark推理结果保存路径
      - --pre_data_save_path：预处理数据保存路径
      - --res_data_save_path：后处理结果保存路径



