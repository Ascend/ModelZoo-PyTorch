# MaskRcnn Onnx模型端到端推理指导

- ATC转换脚本maskrcnn_atc.sh
- get_info.py文件用于处理得到二进制及jpg信息文件
- benchmark二进制文件
- 前处理脚本maskrcnn_pth_preprocess_detectron2.py
- 推理后处理脚本maskrcnn_pth_postprocess_detectron2.py
- ReadMe.md

 [MaskRcnn Onnx模型端到端推理指导](#MaskRcnn-onnx模型端到端推理指导)
	- [1 数据集预处理]
		- [1.1 数据集获取](#1.1-数据集获取)
		- [1.2 数据集预处理](#1.2-数据集预处理)
		- [1.3 生成数据集信息文件](#1.3-生成数据集信息文件)
   -  [2 离线推理]
		- [2.1 安装detectrin2库](#2.1-安装detectrin2库)
		- [2.2 生成onnx文件](#2.2-生成onnx文件)
      - [2.3 ONNX模型转OM模型](#2.3-ONNX模型转OM模型)
      - [2.4 进行推理验证](#2.4-进行推理验证)
	-  [3 精度对比](#3-精度对比)
	-  [4 性能对比](#4-性能对比)
		- [4.1 310性能数据](#4.1-310性能数据)
		- [4.2 310p性能数据](#4.2-310p性能数据)
		- [4.3 性能对比](#4.3-性能对比)

推理端到端步骤：
  ##1. 数据集预处理
    **[数据集获取](1.1-数据集获取)**
    **[数据集预处理](1.2-数据集预处理)**
    **[生成数据集信息文件](1.3-生成数据集信息文件)**
  ###1.1 数据集获取
   datasets/coco目录下有annotations与val2017，annotations目录存放coco数据集的instances_val2017.json，val2017目录存放coco数据集的5000张验证图片。
   在服务器home目录下创建自己的文件夹，并将数据集存放于根目录。

  ###1.2 数据集预处理
   将原始数据（.jpg）转化为二进制文件（.bin）。通过缩放、均值方差手段归一化，输出为二进制文件。
   执行maskrcnn_pth_preprocess_detectron2.py脚本。
   执行以下命令：
   python3.7 maskrcnn_pth_preprocess_detectron2.py --image_src_path=./datasets/coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
   运行成成功后，生成二进制文件val2017_bin。

  ###1.3 生成数据集信息文件
   1.使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。
   执行命令：
   python3.7 get_info.py bin ./val2017_bin maskrcnn.info 1344 1344
   运行成功后，在当前目录中生成maskrcnn.info。
   
   2.JPG图片info文件生成:
   python3.7 get_info.py jpg ./datasets/coco/val2017 maskrcnn_jpeg.info
   运行成功后，在当前目录中生成maskrcnn_jpeg.info。

   ##2. 离线推理
    **[安装detectrin2库](2.1-安装detectron2库)**
    **[生成onnx文件](2.2-生成onnx文件)**
    **[ONNX模型转OM模型](2.3-ONNX模型转OM模型)**
    **[进行推理验证]（2.4-进行推理验证）**
  ###2.1 安装detectron2库
   1.下载代码仓，到ModleZoo获取的源码包根目录下：
   git clone https://github.com/facebookresearch/detectron2到当前文件夹
   cd detectron2/
   git reset --hard 068a93a 

   2.安装detectron2：
   rm -rf detectron2/build/ **/*.so
   python3.7 -m pip install -e detectron2

   3.修改源代码
   patch -p1 < ../maskrcnn_detectron2.diff
   cd ..

   4.找到自己conda环境中的pytorch安装地址
   打开/root/anaconda3/envs/自己创建的环境名称/lib/python3.7/site-packages/torch/onnx/utils.py文件
   搜索_check_onnx_proto(proto)并注释代码，添加pass代码，后保存并退出。
    # _check_onnx_proto(proto)
    pass

  ###2.2 生成onnx文件
   运行命令，在outpu文件夹下生成model.onnx文件
   python3.7 detectron2/tools/deploy/export_model.py --config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --output ./output --export-method tracing --format onnx MODEL.WEIGHTS model_final.pth MODEL.DEVICE cpu
   将其改名：
   mv output/model.onnx 

  ###2.3 ONNX模型转OM模型
   运行atc_crnn.sh脚本将2.2生成的model.onnx文件转换om模型
   1.引用环境变量：source /usr/local/Ascend/ascend-toolkit/set_env.sh

   2.查看卡的型号：npu-smi info

   3.执行命令(请勿直接执行，请根据②中查找的卡型号修改命令)：
      atc --model=model_py1.8.onnx --framework=5 --output=maskrcnn_detectron2_npu --input_format=NCHW --input_shape="0:1,3,1344,1344" --out_nodes="Cast_1673:0;Gather_1676:0;Reshape_1667:0;Slice_1706:0" --log=error --soc_version=Ascend310

   4.执行maskrcnn_atc.sh脚本，将.onnx文件转为离线推理模型文件.om文件：
     bash maskrcnn_atc.sh

  ###2.4 进行推理验证 
     1.增加benchmark.{arch}可执行权限:
       chmod u+x benchmark.x86_64

     2.执行命令，进行推理:
       ./benchmark.x86_64 -model_type=vision -om_path=maskrcnn_detectron2_npu.om -device_id=0 -batch_size=1 -input_text_path=maskrcnn.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
       注意maskrcnn.info为处理后的数据集信息，推理后的输出默认在当前目录result下。

     3.调用后处理脚本maskrcnn_pth_postprocess_detectron2获取txt文件：
       python3.7 maskrcnn_pth_postprocess_detectron2.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=maskrcnn_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=4 --net_input_height=1344 --net_input_width=1344 --ifShowDetObj

   ## 3 精度对比
     
      将得到的om离线模型推理精度与源码包的精度对比，精度更高，故精度达标。  
    **精度调试：**  
>     没有遇到精度不达标的问题，故不需要进行精度调试。

-  ## 4 性能对比

-   **[npu性能数据](#41-310性能数据)**  
-   **[T4性能数据](#42-310P性能数据)**  
-   **[性能对比](#43-性能对比)**  
    
    注意：该模型目前仅支持batchsize=1.

  ### 4.1 310性能数据
      benchmark工具在整个数据集上推理获得性能数据 
      batch1性能：
         
      [e2e] throughputRate: 1.77441, latency: 2.81784e+06
      [data read] throughputRate: 1.83966, moduleLatency: 543.58
      [preprocess] throughputRate: 1.80165, moduleLatency: 555.046
      [infer] throughputRate: 1.77503, Interface throughputRate: 2.00815, moduleLatency: 561.56
      [post] throughputRate: 1.775, moduleLatency: 563.381

      batch1 310单卡吞吐率：2.00815 * 4 = 8.0326fps

   ### 4.2 310p性能数据
      在装有310p卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务
      batch1性能：

      [e2e] throughputRate: 5.74867, latency: 869767
      [data read] throughputRate: 6.02356, moduleLatency: 166.015
      [preprocess] throughputRate: 5.8614, moduleLatency: 170.608
      [infer] throughputRate: 5.7554, Interface throughputRate: 11.4222, moduleLatency: 163.148
      [post] throughputRate: 5.75507, moduleLatency: 173.76

      batch1 310p单卡吞吐率：11.4222

   ### 4.3 性能对比
      batch1： (310)2.00815 * 4 = 8.0326fps < (310p) 11.422fps

      310p上的性能为310性能上的1.2倍以上，性能达标。





