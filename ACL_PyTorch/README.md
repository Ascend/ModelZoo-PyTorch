# 欢迎使用Ascend ACL_PyTorch

为方便更多开发者体验和使用昇腾芯片澎湃推理算力，该目录下提供了经典和主流算法模型实现昇腾服务器推理的端到端流程，更多模型持续更新中。如果您有任何需求，请在[modelzoo/issues](https://gitee.com/ascend/modelzoo/issues)提交issue，我们会及时处理。

## 如何贡献

在开始贡献之前，请先阅读[CONTRIBUTING](https://gitee.com/ascend/modelzoo/blob/master/CONTRIBUTING.md)。
谢谢！

## 支持模型列表（按字母顺序排序）
>**说明：**   
>**以下无精度指标的模型均需人工与在线推理结果比较**  
>**因使用版本差异，模型性能可能存在波动，性能仅供参考**

##  规范模型
CV-classfication

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
	<td>Top1Acc</td>
	<td>Top5Acc</td>
        <td>mAP</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Big-transfer"> Big-transfer </a>
        </td>
        <td>CIFAR-10</td>
	<td>97.62%</td>
        <td></td>
	<td></td>
	<td>1758.00(bs16)</td>
	<td>bs x 3 x 128 x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/C3D">  C3D </a>
        </td>
        <td>UCF101</td>
	<td>81.87%</td>
        <td></td>
	<td></td>
	<td>54.92(bs4)</td>
	<td>bs x 10 x 3 x16 x 112 x 112</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Densenet121_Pytorch_Infer">   Densenet121_Pytorch_Infer </a>
        </td>
        <td>ImageNet</td>
	<td>71.43%</td>
        <td>91.96%</td>
	<td></td>
	<td>2368(bs8)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/GaitSet">   GaitSet </a>
        </td>
        <td>CASIA-B</td>
	<td>95.512%</td>
	<td></td>
        <td></td>
	<td>723(bs64)</td>
	<td>bs x 100 x 64 x 44</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/InceptionV3_for_Pytorch">  InceptionV3_for_Pytorch </a>
        </td>
        <td>ImageNet</td>
	<td>77.31%</td>
	<td>93.46%</td>
        <td></td>
	<td>2736(bs8)</td>
	<td>bs x 3 x 299 x2 99</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MaskRcnn">  MaskRcnn </a>
        </td>
        <td>coco</td>
        <td></td>
        <td></td>
        <td>32.739%</td>
        <td>13(bs1)</td>
        <td>1 x 3 x 1344 x 1344</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MobileNet-v1"> MobileNet-v1 </a>
        </td>
        <td>ImageNet</td>
	<td>69.52%</td>
        <td>89.05%</td>
	<td></td>
	<td>16124.099(bs32)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/MobileNetV2_for_Pytorch"> MobileNetV2_for_Pytorch </a>
        </td>
        <td>ImageNet</td>
	<td>71.87%</td>
        <td>90.32%</td>
	<td></td>
	<td>7072(bs4)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/MobileNetV3_for_Pytorch"> MobileNetV3_for_Pytorch </a>
        </td>
        <td>ImageNet</td>
	<td>65.094%</td>
        <td>85.432%</td>
	<td></td>
	<td>16245(bs32)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/RegNetX-1.6GF"> RegNetX-1.6GF </a>
        </td>
        <td>ImageNet</td>
	<td>76.93%</td>
        <td>93.43%</td>
	<td></td>
	<td>5426.759(bs8)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/RepVGG">   RepVGG </a>
        </td>
        <td>ImageNet</td>
	<td>72.15%</td>
        <td>90.4%</td>
	<td></td>
	<td>8933.6(bs16)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ResNeSt50">  ResNeSt50 </a>
        </td>
        <td>ImageNet</td>
	<td>81%</td>
        <td></td>
	<td></td>
	<td>1704(bs4)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ResNext101_32x8d"> ResNext101_32x8d </a>
        </td>
        <td>ImageNet</td>
	<td>79.312%</td>
        <td>94.526%</td>
	<td></td>
	<td>1251(bs8)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SCNet"> SCNet </a>
        </td>
        <td>ImageNet</td>
	<td>80.34%</td>
        <td></td>
	<td></td>
	<td>2331(bs4)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Shufflenetv1"> Shufflenetv1 </a>
        </td>
        <td>ImageNet</td>
	<td>67.71%</td>
        <td></td>
	<td></td>
	<td>8747(bs16)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Shufflenetv2_for_Pytorch"> Shufflenetv2_for_Pytorch </a>
        </td>
        <td>ImageNet</td>
	<td>69.33%</td>
        <td>88.34%</td>
	<td></td>
	<td>7765(bs32)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Squeezenet1_1">  Squeezenet1_1 </a>
        </td>
        <td>ImageNet</td>
	<td>57.32%</td>
        <td>80.06%</td>
	<td></td>
	<td>21301(bs8)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/VGG16">  VGG16 </a>
        </td>
        <td>ImageNet</td>
	<td>71.28%</td>
        <td>90.38%</td>
	<td></td>
	<td>1508(bs16)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ViT_base"> ViT_base </a>
        </td>
        <td>ImageNet</td>
	<td>80.63%(patch32_224)</td>
        <td></td>
	<td></td>
	<td>1679.63(patch32_224 bs64)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
</table>


CV-detection

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
	<td>AP</td>
        <td>mAP</td>
        <td>Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/CascadeRCNN-DCN"> CascadeRCNN-DCN </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>43.8%</td>
        <td></td>
	<td>3.9(bs1)</td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/CenterFace"> CenterFace </a>
        </td>
        <td>WIDER_FACE</td>
	<td></td>
        <td>91.02%</td>
        <td></td>
	<td>439.9(bs1)</td>
	<td>bs x 3 x 800 x 800</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/CenterNet">  CenterNet </a>
        </td>
        <td>coco</td>
	<td>36.4%</td>
        <td></td>
        <td></td>
	<td>34.1(bs4)</td>
	<td>bs x 3 x 512 x 512 <br> bs x 3 x 800 x 800</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/CRNN_BuiltIn_for_Pytorch">  CRNN_BuiltIn_for_Pytorch </a>
        </td>
        <td>IIIT5K_lmdb</td>
	<td>74.87%</td>
        <td></td>
        <td></td>
	<td>19075(bs64)</td>
	<td>bs x 1 x 32 x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/CRNN_meijieru_for_Pytorch">  CRNN_meijieru_for_Pytorch </a>
        </td>
        <td>demo文件</td>
	<td></td>
        <td></td>
        <td></td>
	<td>19374(bs64)</td>
	<td>bs x 1 x 32 x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/CRNN_Dynamic_for_PyTorch">  CRNN_Dynamic_for_PyTorch </a>
        </td>
        <td>IIIT5K_lmdb</td>
	<td>66.8%</td>
        <td></td>
        <td></td>
	<td>5.9~6.1s(E2E)</td>
	<td>bs x 1 x 32 x len</td>
    </tr>
    <tr>
        <td><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/ocr/CRNN/CRNN_Sierkinhane_for_Pytorch">CRNN_Sierkinhane_for_Pytorch</a></td>
        <td>GitHub 仓库提供的 360 万数据集</td>
	    <td>78.37%</td>
        <td></td>
        <td></td>
	    <td>6011(bs64)</td>
	    <td>bs x 1 x 32 x 160</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/DBNet_MobileNetV3">  DBNet_MobileNetV3 </a>
        </td>
        <td>ICDAR2015</td>
	<td></td>
        <td></td>
        <td>77.5%</td>
	<td>196(bs1)</td>
	<td>bs x 736 x 1280 x 3</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Detr">  Detr </a>
        </td>
        <td>coco</td>
	<td>41.6%</td>
        <td></td>
        <td></td>
	<td>63.75(bs1)</td>
	<td>多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/EfficientDetD0">   EfficientDetD0 </a>
        </td>
        <td>coco</td>
	<td>33.4%</td>
        <td></td>
        <td></td>
	<td>260(bs4)</td>
	<td>bs x 3 x 512 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50">  Faster_R-CNN_ResNet50 </a>
        </td>
        <td>coco</td>
	<td>37.2%</td>
        <td></td>
        <td></td>
	<td>15.7(bs1)</td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/OpenPose"> OpenPose </a>
        </td>
        <td>coco</td>
	<td>40.4%</td>
        <td></td>
        <td></td>
	<td>945.99(bs32)</td>
	<td>bs x 3 x 368 x 6406</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/PSENet_for_Pytorch">  PSENet_for_Pytorch </a>
        </td>
        <td>ICDAR2015</td>
	<td></td>
        <td></td>
        <td>80.5%</td>
	<td>70(bs1)</td>
	<td>bs x 3 x 704 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Retinanet_for_Pytorch"> Retinanet_r50_fpn_1x </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>36.3%</td>
        <td></td>
	<td>15.48(bs1)</td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Retinanet_Resnet18"> Retinanet_Resnet18 </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>31.6%</td>
        <td></td>
	<td>16.78(bs1)</td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/SSD"> SSD </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>25.4%</td>
        <td></td>
	<td>337.01(bs4)</td>
	<td>bs x 3 x 300 x 300</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/SSD-Resnet34"> SSD-Resnet34 </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>23%</td>
        <td></td>
	<td>1550(bs8)</td>
	<td>bs x 3 x 300 x 300</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov3_for_PyTorch"> Yolov3_for_PyTorch </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>63.3%</td>
        <td></td>
	<td>219(bs4)</td>
	<td>bs x 3 x 640 x 640</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Yolov4_for_Pytorch"> Yolov4_for_Pytorch </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>60.3%</td>
        <td></td>
	<td>178(bs4)</td>
	<td>bs x 3 x 416 x 416</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch"> Yolov5_for_Pytorch </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>55.3%</td>
        <td></td>
	<td>1079(bs4)</td>
	<td>bs x 3 x 640 x 640</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/YOLOX"> YOLOX </a>
        </td>
        <td>coco</td>
	<td>51.2%</td>
        <td></td>
        <td></td>
	<td>77.4(bs64)</td>
	<td>bs x 3 x 640 x 640</td>
</table>

CV-segmentation

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=2>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>Acc</td>
        <td>mAP</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/DB_for_PyTorch"> DB_for_PyTorch </a>
        </td>
        <td>icdar2015</td>
        <td></td>
	<td>88%</td>
	<td>19(bs16)</td>
	<td>bs x 3 x 736 x 1280</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/DB_Dynamic_for_PyTorch"> DB_Dynamic_for_PyTorch </a>
        </td>
        <td>icdar2015</td>
        <td></td>
	<td></td>
	<td>E2E: 9.3s</td>
	<td>1 x 3 x h x w</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/DeeplabV3"> DeeplabV3 </a>
        </td>
        <td>Cityscapes</td>
	<td>79.12%</td>
        <td></td>
	<td>4.9(bs1)</td>
	<td>1 x 3 x 1024 x 2048</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/SOLOV2">  SOLOV2 </a>
        </td>
        <td>coco</td>
	<td>34%</td>
        <td></td>
	<td>17(bs1)</td>
	<td>1 x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/UNet">  UNet </a>
        </td>
        <td>carvana</td>
	<td>98.6%</td>
        <td></td>
	<td>78(bs1)</td>
	<td>bs x 3 x 572 x 572</td>
</table>

CV-gan

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" rowspan=2>精度</th>
	<th colspan=2>最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>   
        <td>310P</td>
        <td>310</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/StarGAN">  StarGAN </a>
        </td>
        <td>celeba</td>
        <td></td>
	<td>1281(bs8)</td>
        <td></td>
	<td nowrap="nowrap">bs x 3 x 128 x 128 <br> bs x 5</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/StyleGAN2-ADA">   StyleGAN2-ADA </a>
        </td>
        <td>代码仓提供</td>
        <td></td>
	<td></td>
        <td>19(bs1)</td>
	<td nowrap="nowrap">1 x 512</td>
</table>

CV-pose_estimation

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=2>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td nowrap="nowrap">Top1Acc</td>
        <td>Top5Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/HRNet"> HRNet </a>
        </td>
        <td>ImageNet</td>
	<td>76.45%</td>
        <td>93.14%</td>
	<td>1673(bs16)</td>
	<td nowrap="nowrap">bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/STGCN_for_Pytorch"> STGCN_for_Pytorch </a>
        </td>
        <td>Kinetics</td>
	<td>31.59%</td>
        <td>53.74%</td>
	<td>381(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 300 x 18 x 2</td>
</table>

CV-super_resolution

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=1>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td nowrap="nowrap">PSNR</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/super_resolution/EDSR">  EDSR </a>
        </td>
        <td>DIV2K</td>
	<td>34.6</td>
	<td>7.9(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 1020 x 1020</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/EDSR_Dynamic_for_PyTorch">  EDSR_Dynamic_for_PyTorch </a>
        </td>
        <td>B100</td>
	<td>32.35</td>
	<td>6.9(H:240, W:320)</td>
	<td nowrap="nowrap">多尺度</td>
</table>

CV-tracking


<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=1>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td nowrap="nowrap">Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/tracking/Deepsort_for_Pytorch">  Deepsort_for_Pytorch </a>
        </td>
        <td>MOT16</td>
	<td>30.1%</td>
	<td>yolov3:467(bs1) <br> deep:2950(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 416 x 416</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/T2vec_for_Pytorch">  T2vec_for_PyTorch </a>
        </td>
        <td>Proto</td>
	<td>/</td>
	<td>9.85ms</td>
	<td nowrap="nowrap">动态输入</td>
</table>

CV-image_registration
<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=2>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td nowrap="nowrap">Auc@20</td>
        <td>Precision</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SuperGlue_with_SuperPoint_for_Pytorch"> SuperGlue_with_SuperPoint_for_Pytorch </a>
        </td>
        <td>YFCC100M</td>
	<td>75.04%</td>
        <td>97.85%</td>
	<td>31.10s(e2e)</td>
	<td nowrap="nowrap">1 x 1 x 1200 x 1600</td>
</table>

CV-video_understanding

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=2>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td nowrap="nowrap">Top1Acc</td>
        <td>Top5Acc</td>
    </tr>
    <tr>
        <td>
        <a href=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/TSM_sthv2_for_Pytorch> TSM_sthv2_for_Pytorch </a>
        </td>
        <td>sthv2</td>
	<td>61.87%</td>
        <td>87.21%</td>
	<td>20(bs1)</td>
	<td nowrap="nowrap">bs x 48 x 3 x 256 x 256</td>
</table>

Audio

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td nowrap="nowrap">WER</td>
        <td>CER</td>
        <td>Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/Deepspeech2">   Deepspeech2 </a>
        </td>
        <td>an4</td>
	<td>9.573</td>
        <td>5.515</td>
        <td></td>
	<td>7.74(bs32)</td>
	<td nowrap="nowrap">bs x 1 x 161 x 621 <br> bs x 1</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/EspNet_for_Pytoch">    EspNet_for_Pytoch </a>
        </td>
        <td>代码仓提供</td>
	<td></td>
        <td></td>
        <td></td>
	<td>430(分档)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/FastSpeech2_for_PyTorch">  FastSpeech2_for_PyTorch </a>
        </td>
        <td>LJSpeech</td>
	<td></td>
        <td></td>
        <td></td>
	<td>13.66(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/HiFiGAN_for_PyTorch">  HiFiGAN_for_PyTorch </a>
        </td>
        <td>LJSpeech</td>
	<td></td>
        <td></td>
        <td></td>
	<td>637(bs8 mel_len:250)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/Tacotron2">   Tacotron2 </a>
        </td>
        <td>LJSpeech</td>
	<td></td>
        <td></td>
        <td></td>
	<td>33508(bs16)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/TDNN_for_Pytorch">  TDNN_for_Pytorch </a>
        </td>
        <td>Mini Librispeech</td>
	<td></td>
        <td></td>
        <td>99.93%</td>
	<td>1682(bs64)</td>
	<td nowrap="nowrap">bs x 1800 x 24</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/Conformer_for_Pytorch">  Conformer_for_Pytorch </a>
        </td>
        <td>aishell</td>
	<td></td>
        <td></td>
        <td>95.04%</td>
	<td>60</td>
	<td nowrap="nowrap">多尺度</td>
</table>

Nlp

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=1>精度</th>
	<th rowspan=2>310P最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/albert">  albert </a>
        </td>
        <td>SST-2</td>
        <td>92.8%</td>
	<td>1327(bs8)</td>
	<td nowrap="nowrap">bs x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Bert_Base_Cased_SST2">  Bert_Base_Cased_SST2 </a>
        </td>
        <td>SST-2</td>
        <td>92.43%</td>
	<td>2906(bs32)</td>
	<td nowrap="nowrap">bs x 128 <br> bs x 64</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch">  Bert_Base_Chinese_for_Pytorch </a>
        </td>
        <td>zhwiki</td>
        <td>77.94%</td>
	<td>254(bs8)</td>
	<td nowrap="nowrap">bs x 384</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Bert_Uncased_Huggingface">  Bert_Uncased_Huggingface </a>
        </td>
        <td>SQuAD 1.1</td>
        <td>88.2%</td>
	<td>328.64(bs4)</td>
	<td nowrap="nowrap">bs x 384</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Ernie3_for_Pytorch">  Ernie3_for_Pytorch </a>
        </td>
        <td>Clue</td>
        <td>49%</td>
	<td>1313(bs8)</td>
	<td nowrap="nowrap">bs x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/m2m_100">  m2m_100 </a>
        </td>
        <td>sacrebleu</td>
        <td></td>
	<td>35(bs1)</td>
	<td nowrap="nowrap">1 x 90</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/roberta">  roberta </a>
        </td>
        <td>SST-2</td>
        <td>94.7%</td>
	<td>996(bs64)</td>
	<td nowrap="nowrap">bs x 70</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/SAST">  SAST </a>
        </td>
        <td>ICDAR</td>
        <td>91.3%</td>
	<td>22(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 896 x 1536</td>
</tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/PPOCRV2.5_det_db_mv3">  PPOCRV2.5_det_db_mv3 </a>
        </td>
        <td>icdar2015</td>
        <td>75%</td>
	<td>172(bs24)</td>
	<td nowrap="nowrap">24 x 3 x 736 x 1280</td>
    </tr>
    <tr>
        <td><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/BiLSTM_CRF_PyTorch">  BiLSTM_CRF_PyTorch </a></td>
        <td> CLUE_NER </td>
        <td> f1=0.714 </td>
        <td> 961(bs32)</td>
        <td> ids:bs,50;mask:bs,50 </td>
    </tr>
</table>





##  生态贡献模型
CV-classfication

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th colspan=2>最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
	<td>Top1Acc</td>
	<td>Top5Acc</td>
        <td>mAP</td>
        <td>310P</td>
        <td>310</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/3d_attention_net"> 3d_attention_net </a>
        </td>
        <td>CIFAR-10</td>
	<td>62.2%</td>
	<td></td>
        <td></td>
	<td>7806.96(bs16)</td>
        <td></td>
	<td>bs x 3 x 32 x 32</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/3d_resnets"> 3d_resnets </a>
        </td>
        <td>hmdb51</td>
	<td>62.22%</td>
	<td></td>
        <td></td>
	<td>830.7165(bs10)</td>
        <td></td>
	<td>10 x 3 x 16 x 112 x 112</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/AlexNet"> AlexNet </a>
        </td>
        <td>ImageNet</td>
	<td>56.56%</td>
	<td>79.1%</td>
        <td></td>
	<td>12672(bs64)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/baseline_dino_resnet50"> baseline_dino_resnet50 </a>
        </td>
        <td>ImageNet</td>
	<td>75.28%</td>
	<td>92.56%</td>
        <td></td>
	<td>87537(bs64)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/BMN">  BMN </a>
        </td>
        <td>Activity1.3</td>
	<td>67.69%</td>
        <td></td>
	<td></td>
	<td>114.34(bs1)</td>
        <td></td>
	<td>bs x 400 x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Beit">  Beit </a>
        </td>
        <td>ImageNet</td>
	<td>84.68%</td>
        <td></td>
        <td></td>
	<td>516.00(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ch_ppocr_mobile_v2.0_cls">  ch_ppocr_mobile_v2.0_cls </a>
        </td>
        <td>PaddleOCR</td>
	<td></td>
        <td></td>
	<td></td>
	<td>31958.773(bs64)</td>
        <td></td>
	<td>bs x 3 x 48 x 192</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Conformer-base">   Conformer-base </a>
        </td>
        <td>ImageNet</td>
	<td>83.85%</td>
	<td></td>
        <td></td>
	<td>257.7437(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Conformer_Ti">  Conformer_Ti </a>
        </td>
        <td>ImageNet</td>
	<td>81.09%</td>
        <td></td>
        <td></td>
	<td>907.58(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/convmixer_1536_20">  convmixer_1536_20 </a>
        </td>
        <td>ImageNet</td>
	<td>81.37%</td>
        <td></td>
        <td></td>
	<td>102.9(bs1)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ConvNext_for_Pytorch">  ConvNext_for_Pytorch </a>
        </td>
        <td>ImageNet</td>
	<td>82.094%</td>
        <td></td>
        <td></td>
	<td>461.9(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/CSPResneXt50">  CSPResneXt50 </a>
        </td>
        <td>ImageNet</td>
	<td>79.79%</td>
        <td></td>
        <td></td>
	<td>3251(bs4)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/CSWin-Transformer">  CSWin-Transformer </a>
        </td>
        <td>ImageNet</td>
	<td>83.3%</td>
        <td></td>
        <td></td>
	<td>223.5(bs16)</td>
         <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Deit_Small">   Deit_Small </a>
        </td>
        <td>ImageNet</td>
        <td>79.5%</td>
        <td>94.83%</td>
        <td></td>
        <td></td>
        <td>415(bs1)</td>
        <td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>    
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/DPN131">  DPN131 </a>
        </td>
        <td>ImageNet</td>
	<td>79.47%</td>
	<td>94.54%</td>
        <td></td>
	<td>550(bs4)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Efficient-3DCNNs">   Efficient-3DCNNs </a>
        </td>
        <td>UCF-101</td>
	<td>81.073%</td>
	<td>96.325%</td>
        <td></td>
	<td>1245.1167(bs4)</td>
        <td></td>
	<td>bs x 3 x 16 x 112 x 112</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/EfficientNet-B1">    EfficientNet-B1 </a>
        </td>
        <td>ImageNet</td>
	<td>75.940%</td>
	<td>92.774%</td>
        <td></td>
	<td>1409.692(bs8)</td>
        <td></td>
	<td>bs x 3 x 240 x 240</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/EfficientNet-B3">   EfficientNet-B3 </a>
        </td>
        <td>ImageNet</td>
	<td>76.25%</td>
	<td>92.56%</td>
        <td></td>
	<td>739.02(bs16)</td>
        <td></td>
	<td>bs x 3 x 300 x 300</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/EfficientNet-B5">  EfficientNet-B5 </a>
        </td>
        <td>ImageNet</td>
	<td>77.2%</td>
	<td>92.8%</td>
        <td></td>
	<td>166.408(bs64)</td>
        <td></td>
	<td>bs x 3 x 456 x 456</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/EfficientNet_b7_for_Pytorch"> EfficientNet_b7_for_Pytorch </a>
        </td>
        <td>ImageNet</td>
	<td>84.4%</td>
	<td></td>
        <td></td>
	<td>75.9(bs32)</td>
        <td></td>
	<td>bs x 3 x 600 x 600</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/EfficientNetV2_for_Pytorch"> EfficientNetV2_for_Pytorch </a>
        </td>
        <td>ImageNet</td>
	<td>82.26%</td>
	<td></td>
        <td></td>
	<td>1670(bs64)</td>
        <td></td>
	<td>bs x 3 x 288 x 288</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/FixRes">   FixRes </a>
        </td>
        <td>ImageNet</td>
	<td>79.0%</td>
	<td></td>
        <td></td>
	<td>984(bs4)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/FocalTransformer">  FocalTransformer </a>
        </td>
        <td>ImageNet</td>
	<td>83.586%</td>
	<td></td>
        <td></td>
	<td>7.96(bs1)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/GENet">   GENet </a>
        </td>
        <td>CIFAR-10</td>
	<td>94.23%</td>
	<td></td>
        <td></td>
	<td>9981(bs16)</td>
        <td></td>
	<td>bs x 3 x 32 x 32</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/GhostNet1.0x">    GhostNet1.0x </a>
        </td>
        <td>ImageNet</td>
	<td>73.98%</td>
	<td></td>
        <td></td>
	<td>4974(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/GloRe">   GloRe </a>
        </td>
        <td>UCF101</td>
	<td>92.12%</td>
	<td>99.56%</td>
        <td></td>
	<td>85(bs4)</td>
        <td></td>
	<td>bs x 3 x 8 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/GoogleNet_for_Pytorch">  GoogleNet </a>
        </td>
        <td>ImageNet</td>
	<td>69.78%</td>
	<td>89.53%</td>
        <td></td>
	<td>6308.38(bs8)</td>
        <td></td>
	<td>bs x 3 x 8 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/HRNet-Image-Classification">  HRNet-Image-Classification </a>
        </td>
        <td>ImageNet</td>
	<td>76.51%</td>
	<td>93.22%</td>
        <td></td>
	<td>2250(bs16)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/InceptionResnetV2">  InceptionResnetV2 </a>
        </td>
        <td>ImageNet</td>
	<td>80.15%</td>
	<td>95.24%</td>
        <td></td>
	<td>1310.5(bs8)</td>
        <td></td>
	<td>bs x 3 x 299 x 299</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Inception_V4">  Inception_V4 </a>
        </td>
        <td>ImageNet</td>
	<td>79.99%</td>
	<td>94.86%</td>
        <td></td>
	<td>1498.5(bs4)</td>
        <td></td>
	<td>bs x 3 x 299 x 299</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/LResNet100E-IR">   LResNet100E-IR </a>
        </td>
        <td>LFW</td>
        <td>99.7%</td>
        <td></td>
        <td></td>
        <td></td>
        <td>746(bs16)</td>
        <td>bs x 3 x 112 x 112</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/LV-Vit">  LV-Vit </a>
        </td>
        <td>ImageNet</td>
        <td>83.3%</td>
        <td></td>
        <td></td>
        <td>407.14(bs8)</td>
        <td></td>
        <td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MAE"> MAE </a>
        </td>
        <td>ImageNet</td>
	<td>83.52%%</td>
        <td></td>
        <td></td>
	<td>266.8(bs1)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Mnasnet1_0"> Mnasnet1_0 </a>
        </td>
        <td>ImageNet</td>
	<td>73.48%</td>
        <td></td>
        <td></td>
	<td>10650.988(bs16)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MobileNetV3_large_100"> MobileNetV3_large_100 </a>
        </td>
        <td>ImageNet</td>
	<td>75.62%</td>
        <td>92.47%</td>
	<td></td>
	<td>6998.89(bs16)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Moco-v2"> Moco-v2 </a>
        </td>
        <td>ImageNet</td>
        <td>67.28%</td>
        <td>87.82%</td>
        <td></td>
        <td>3288.38(bs4)</td>
        <td></td>
        <td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/OSNet"> OSNet </a>
        </td>
        <td>Market-1501</td>
	<td></td>
        <td></td>
	<td>82.55%</td>
	<td>4075(bs8)</td>
        <td></td>
	<td>bs x 3 x 256 x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/PAMTRI"> PAMTRI </a>
        </td>
        <td>veri</td>
	<td></td>
        <td></td>
	<td>68.64%</td>
	<td>1564.274(bs4)</td>
        <td></td>
	<td>bs x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/pnasnet5large"> pnasnet5large </a>
        </td>
        <td>ImageNet</td>
	<td>81.76%</td>
        <td></td>
	<td></td>
	<td>203.256(bs4)</td>
        <td></td>
	<td>bs x 3 x 331 x 331</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/PointNet"> PointNet </a>
        </td>
        <td>shapenetcore</td>
	<td>97.35%</td>
        <td></td>
	<td></td>
	<td>2374.11(bs1)</td>
        <td></td>
	<td>bs x 3 x 2500</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/PointNetCNN">  PointNetCNN </a>
        </td>
        <td>modelnet40</td>
	<td>82.82%</td>
        <td></td>
	<td></td>
	<td>273(bs1)</td>
        <td></td>
	<td>1 x 1024 x 3</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Pointnetplus/Pointnetplus">  Pointnetplus </a>
        </td>
        <td>modelnet40</td>
	<td>88.4%</td>
        <td></td>
	<td></td>
	<td>partone7825(bs4)  parttwo5127(bs1)</td>
        <td></td>
	<td nowrap="nowrap">partone[bs, 512, 3] [bs, 3, 32, 512] <br> parttwo[bs, 3, 128] [bs, 131, 64, 128]</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/R(2+1)D"> R(2+1)D </a>
        </td>
        <td>UCF-101</td>
	<td>89.23%</td>
        <td>97.45%</td>
	<td></td>
	<td>84.7777(bs32)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ReID_for_Pytorch"> ReID_for_Pytorch </a>
        </td>
        <td>Market1501</td>
	<td></td>
        <td></td>
	<td>85.9%</td>
	<td>4417.628(bs16)</td>
        <td></td>
	<td>bs x 3 x 256 x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/RegNetY-1.6GF">  RegNetY-1.6GF </a>
        </td>
        <td>ImageNet</td>
	<td>77.86%</td>
        <td>93.72%</td>
	<td></td>
	<td>4417.628(bs16)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Res2Net101_v1b">Res2Net101_v1b </a>
        </td>
        <td>ImageNet</td>
        <td>81.22%</td>
        <td>95.36%</td>
        <td></td>
        <td></td>
        <td>347(bs32)</td>
        <td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ResNet152"> ResNet152 </a>
        </td>
        <td>ImageNet</td>
	<td>78.31%</td>
        <td>94.05%</td>
	<td></td>
	<td>1844(bs8)</td>
        <td></td>
	<td>bs x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ResNet34"> ResNet34 </a>
        </td>
        <td>ImageNet</td>
	<td>73.31%</td>
        <td>91.44%</td>
	<td></td>
	<td>5455(bs16)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/resnet50_mmlab_for_pytorch">  resnet50_mmlab_for_pytorch </a>
        </td>
        <td>cifar100</td>
	<td>79.9%</td>
        <td></td>
	<td></td>
	<td>9329(bs16)</td>
        <td></td>
	<td>bs x 3 x 32 x 32</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet18_for_PyTorch">  Resnet18_for_PyTorch </a>
        </td>
        <td>ImageNet</td>
	<td>69.75%</td>
        <td>89.10%</td>
	<td></td>
	<td>9828(bs128)</td>
        <td></td>
	<td>bs x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer">  Resnet50_Pytorch_Infer </a>
        </td>
        <td>ImageNet</td>
	<td>76.14%</td>
        <td>92.87%</td>
	<td></td>
	<td>4250(bs64)</td>
        <td></td>
	<td>bs x 3 x 256 x 256</td>
    </tr>
        <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_mlperf">  Resnet50_mlperf </a>
        </td>
        <td>ImageNet</td>
    <td>76.44%</td>
    <td></td>
    <td>3940.45(bs64)</td>
        <td></td>
    <td>bs x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet101_Pytorch_Infer">  ResNet101_Pytorch_Infer </a>
        </td>
        <td>ImageNet</td>
	<td>77.38%</td>
        <td>93.56%</td>
	<td></td>
        <td>2548(bs8)</td>
	<td></td>
	<td>bs x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Resnetvd">  Resnetvd </a>
        </td>
        <td>ImageNet</td>
	<td>77.37%</td>
        <td>93.77%</td>
	<td></td>
	<td>2823(bs32)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ResNeXt50"> ResNeXt50 </a>
        </td>
        <td>ImageNet</td>
	<td>77.61%</td>
        <td></td>
	<td></td>
	<td>3985(bs32)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/SENet"> SENet </a>
        </td>
        <td>ImageNet</td>
	<td>77.64%</td>
        <td>93.74%</td>
	<td></td>
	<td>2479(bs32)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Se-Resnext101"> Se-Resnext101 </a>
        </td>
        <td>ImageNet</td>
	<td>78.24%</td>
        <td></td>
	<td></td>
	<td>927.09(bs4)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SE_ResNet50_Pytorch_Infer"> SE_ResNet50_Pytorch_Infer </a>
        </td>
        <td>ImageNet</td>
	<td>77.36%</td>
        <td>93.76%</td>
	<td></td>
	<td>2690(bs32)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/SE-ResNeXt50_32x4d">  SE-ResNeXt50_32x4d </a>
        </td>
        <td>ImageNet</td>
	<td>79.06%</td>
        <td>94.44%</td>
	<td></td>
	<td>1804.86(bs4)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ShiftViT">  ShiftViT </a>
        </td>
        <td>ImageNet</td>
	<td>79.3%</td>
        <td></td>
	<td></td>
	<td>842.25(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Shufflenetv2+">   Shufflenetv2+ </a>
        </td>
        <td>ImageNet</td>
	<td>74.08%</td>
        <td>91.67%</td>
	<td></td>
	<td>3595.13(bs32)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/SimCLR_inference">   SimCLR_inference </a>
        </td>
        <td>CIFAR-10</td>
	<td>65.55%</td>
        <td></td>
	<td></td>
	<td>28070(bs32)</td>
        <td></td>
	<td>bs x 3 x 32 x 32</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Sknet50">    Sknet50 </a>
        </td>
        <td>ImageNet</td>
	<td>77.54%</td>
        <td></td>
	<td></td>
	<td>2416(bs8)</td>
        <td></td>
	<td>bs x 3 x 32 x 32</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/sMLP">    sMLP </a>
        </td>
        <td>ImageNet</td>
	<td>81.25%</td>
        <td></td>
	<td></td>
	<td>298.7(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/SPACH">  SPACH </a>
        </td>
        <td>ImageNet</td>
	<td>81.5%</td>
        <td></td>
	<td></td>
	<td>462.96(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/spnasnet_100">  spnasnet_100 </a>
        </td>
        <td>ImageNet</td>
	<td>74.19%</td>
        <td>91.95%</td>
	<td></td>
	<td>8408(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SwinTransformer_for_Pytorch"> SwinTransformer_for_Pytorch </a>
        </td>
        <td>ImageNet</td>
	<td>86.4%</td>
        <td>98%</td>
	<td></td>
	<td>132(bs8)</td>
        <td></td>
	<td>bs x 3 x 384 x 384</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Swin-Transformer_tiny">  Swin-Transformer_tiny </a>
        </td>
        <td>ImageNet</td>
	<td>81.15%</td>
        <td>95.42%</td>
	<td></td>
	<td>564.7(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/T2T-ViT">  T2T-ViT </a>
        </td>
        <td>ImageNet</td>
	<td>81.4%</td>
        <td></td>
	<td></td>
	<td>194.66(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/TimeSformer">   TimeSformer </a>
        </td>
        <td>kinetics400</td>
	<td>77.68%</td>
        <td></td>
	<td></td>
	<td>7.53(bs1)</td>
        <td></td>
	<td>1 x 3 x 3 x 8 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/TNT">  TNT </a>
        </td>
        <td>ImageNet</td>
	<td>81.5%</td>
        <td></td>
	<td></td>
	<td>274(bs8)</td>
        <td></td>
	<td>bs x 196 x 16 x 24</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/TResNet">  TResNet </a>
        </td>
        <td>ImageNet</td>
	<td></td>
        <td>94.43%</td>
	<td></td>
	<td>3249(bs16)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Twins-PCPVT-S">  Twins-PCPVT-S </a>
        </td>
        <td>ImageNet</td>
	<td>81.22%</td>
        <td></td>
	<td></td>
	<td>613(bs16)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Twins-SVT-L">  Twins-SVT-L </a>
        </td>
        <td>ImageNet</td>
	<td>83.7%</td>
        <td></td>
	<td></td>
	<td>175.2209(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/VAN"> VAN </a>
        </td>
        <td>ImageNet</td>
	<td>82.78%</td>
        <td></td>
	<td></td>
	<td>874(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/VGG19"> VGG19 </a>
        </td>
        <td>ImageNet</td>
	<td>71.76%</td>
        <td>90.80%</td>
	<td></td>
	<td>1153(bs64)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Video-Swin-Transformer">  Video-Swin-Transformer </a>
        </td>
        <td>kinetics400</td>
	<td>80.6%</td>
        <td>94.5%</td>
	<td></td>
	<td>0.607(bs1)</td>
        <td></td>
	<td>1 x 12 x 3 x 32 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/vit-small">  vit-small </a>
        </td>
        <td>ImageNet</td>
	<td>81.37%</td>
        <td></td>
	<td></td>
	<td>1013(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/volo"> volo </a>
        </td>
        <td>ImageNet</td>
	<td>82.53%</td>
        <td></td>
	<td></td>
	<td>124(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/vovnet39">  vovnet39 </a>
        </td>
        <td>ImageNet</td>
	<td>76.77%</td>
        <td>93.43%</td>
	<td></td>
	<td>1767(bs4)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Wide_ResNet101_2"> Wide_ResNet101_2 </a>
        </td>
        <td>ImageNet</td>
	<td>78.86%</td>
        <td>94.29%</td>
	<td></td>
	<td>1151(bs16)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Wide_ResNet50_2_for_Pytorch"> Wide_ResNet50_2 </a>
        </td>
        <td>ImageNet</td>
	<td>78.48%</td>
        <td>94.09%</td>
	<td></td>
	<td>2097(bs32)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/xception">  xception </a>
        </td>
        <td>ImageNet</td>
	<td>78.8%</td>
        <td>94.2%</td>
	<td></td>
        <td></td>
	<td>813(bs8)</td>
	<td>bs x 3 x 299 x 299</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/xcit">  xcit </a>
        </td>
        <td>ImageNet</td>
	<td>81.86%</td>
        <td></td>
	<td></td>
	<td>443(bs8)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
</table>

CV-detection

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=7>精度</th>
	<th colspan=2>最优性能(对应bs)</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
	<td>AP</td>
        <td>mAP</td>
        <td nowrap="nowrap">DSC-score</td>
        <td nowrap="nowrap">F1-score</td>
        <td>Top1Acc</td>
        <td>ODS</td>
        <td>loss</td>
        <td>310P</td>
        <td>310</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/3DUnet"> 3DUnet </a>
        </td>
        <td>Brats2018</td>
	<td></td>
        <td></td>
        <td>25.6%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>10.75(bs1)</td>
        <td nowrap="nowrap">bs x 4 x 64 x 64 x 64</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/AdvancedEAST"> AdvancedEAST </a>
        </td>
        <td>天池ICPR</td>
	<td></td>
        <td></td>
        <td></td>
        <td>52.08%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>137(bs1)</td>
        <td></td>
	<td>bs x 3 x 736 x 736</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/AlphaPose"> AlphaPose </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>71.47%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>1772(bs16)</td>
        <td></td>
	<td>bs x 3 x 256 x 192</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/BSN"> BSN </a>
        </td>
        <td>Activity1.3</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>74.34%</td>
        <td></td>
        <td></td>
	<td>34617(bs16)</td>
        <td></td>
	<td nowrap="nowrap">TEM[bs, 400, 100] <br> PEM[bs, 3, 100]</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Cascade_Mask_Rcnn_SwinS"> Cascade_Mask_Rcnn_SwinS </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>51.4%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>3.17(bs1)</td>
        <td></td>
	<td>1 x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/CascadeRCNN-DCN-101_for_Pytorch">  CascadeRCNN-DCN-101_for_Pytorch </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>45%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>3(bs1)</td>
        <td></td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Cascade_RCNN_R101_FPN">  Cascade_RCNN_R101_FPN </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>41.9%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>9(bs1)</td>
        <td></td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Cascade-RCNN-Resnet101-FPN-DCN">  Cascade-RCNN-Resnet101-FPN-DCN </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>45%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>3.8(bs1)</td>
        <td></td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Cascade-RCNN-Resnet50-FPN"> Cascade-RCNN-Resnet50-FPN </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>40.5%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>6.5(bs1)</td>
        <td></td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/ch_ppocr_server_v2.0_det"> ch_ppocr_server_v2.0_det </a>
        </td>
        <td>PaddleOCR</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>154(bs1)</td>
        <td></td>
	<td>多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/ch_PP-OCRv2_det">  ch_PP-OCRv2_det </a>
        </td>
        <td>PaddleOCR</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>232(bs1)</td>
        <td></td>
	<td>多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ch_PP-OCRv3_det">  ch_PP-OCRv3_det</a>
        </td>
        <td>PaddleOCR</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>215(bs1)</td>
        <td></td>
	<td>1 x 3 x -1 x -1</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/CRAFT_for_Pytorch">   CRAFT_for_Pytorch </a>
        </td>
        <td>随机数</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>132(bs1)</td>
        <td></td>
	<td>1 x 3 x 640 x 640</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/CTPN">  CTPN </a>
        </td>
        <td>ICDAR2013</td>
	<td>86.84%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>169(bs64)</td>
        <td></td>
	<td>多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Deepmar_for_Pytorch">  Deepmar_for_Pytorch </a>
        </td>
        <td>PETA</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>78.9%</td>
        <td></td>
        <td></td>
	<td>1642(bs1)</td>
        <td></td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/EAST_MobileNetV3">   EAST_MobileNetV3 </a>
        </td>
        <td>ICDAR2015</td>
	<td>78.29%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>458(bs1)</td>
        <td></td>
	<td>bs x 3 x 704 x 1280</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/EAST_ResNet50_vd">  EAST_ResNet50_vd </a>
        </td>
        <td>ICDAR2015</td>
	<td>88.63%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>91(bs1)</td>
        <td></td>
	<td>bs x 3 x 704 x 1280</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/EfficientDetD7"> EfficientDetD7 </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>53%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>6.18(bs1)</td>
        <td></td>
	<td>bs x 3 x 1536x 1536</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/en_PP-OCRv3_det">  en_PP-OCRv3_det </a>
        </td>
        <td>PaddleOCR</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>168(bs1)</td>
        <td></td>
	<td>多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/FairMOT">   FairMOT </a>
        </td>
        <td>MOT17</td>
	<td>83.7%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>12(bs32)</td>
        <td></td>
	<td>bs x 3 x 608 x 1088</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_DCN_Res101">   Faster_R-CNN_DCN_Res101 </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>44.2%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>2.47(bs1)</td>
        <td></td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_DCN_Res50">   Faster_R-CNN_DCN_Res50 </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>41.1%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>8(bs1)</td>
        <td></td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50">  Faster_R-CNN_ResNet50 </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>37.2%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>15(bs1)</td>
        <td></td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/FCENet">  FCENet </a>
        </td>
        <td>icdar2015</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>87.2%</td>
        <td></td>
        <td></td>
	<td>28.9(bs1)</td>
        <td></td>
        <td nowrap="nowrap">bs x 3 x 1280 x 2272</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Fcos">  Fcos </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>35.9%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>65(bs4)</td>
        <td></td>
	<td>bs x 3 x 800 x 1333</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/FOTS"> FOTS </a>
        </td>
        <td>ICDAR2015</td>
	<td>86.4%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>66(bs16)</td>
        <td></td>
        <td nowrap="nowrap">bs x 3 x1248 x 2240</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Fsaf"> Fsaf </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>37.1%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>20(bs1)</td>
        <td></td>
	<td>bs x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/GFocalV2"> GFocalV2 </a>
        </td>
        <td>coco</td>
	<td>40.6%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>41.8(bs4)</td>
        <td></td>
	<td>bs x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/M2Det"> M2Det </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>37.8%</td>
        <td></td>
        <td></td>
	<td>65(bs4)</td>
        <td></td>
	<td>bs x 3 x 512 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/NAS_FPN">  NAS_FPN </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>40.4%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>72(bs1)</td>
        <td></td>
	<td>1 x 3 x 640 x 640</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Nasnetlarge">  Nasnetlarge </a>
        </td>
        <td>ImageNet</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>82.5%</td>
        <td></td>
        <td></td>
	<td>175(bs4)</td>
        <td></td>
 	<td>bs x 3 x 331x 331</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Pelee_for_Pytorch">  Pelee_for_Pytorch </a>
        </td>
        <td>VOC</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td></td>
        <td></td>
 	<td>bs x 3 x 304 x 304</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/PSE_MobileNetV3">  PSE_MobileNetV3 </a>
        </td>
        <td>ICDAR2015</td>
	<td>82.14%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>219(bs1)</td>
        <td></td>
        <td nowrap="nowrap">bs x 3 x 736 x 1312</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/PSENet_ResNet50_vd">  PSENet_ResNet50_vd </a>
        </td>
        <td>ICDAR2015</td>
	<td>85.72%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>51(bs1)</td>
        <td></td>
	<td>bs x 3 x 736 x 1312</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/pyramidbox"> pyramidbox </a>
        </td>
        <td>widerface</td>
	<td>95%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>8.9(bs1)</td>
        <td></td>
	<td>1 x 3 x 1000 x 1000</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/RCF"> RCF </a>
        </td>
        <td>BSDS500</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>79.8%</td>
        <td></td>
        <td></td>
        <td>93(bs1)</td>
        <td nowrap="nowrap">bs x 3 x 321 x 481 <br> bs x 3 x 481 x 321</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/RefineDet">  RefineDet </a>
        </td>
        <td>VOC2007</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>79.6%</td>
        <td></td>
        <td></td>
	<td>445(bs16)</td>
        <td></td>
        <td>bs x 3 x 320 x 320</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/RetinaMask"> RetinaMask </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>27.9%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>4.3(bs1)</td>
        <td></td>
        <td>1 x 3 x 1344 X 1244</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet"> Retinanet </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>38.3%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>17(bs1)</td>
        <td></td>
	<td>1 x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/RFCN"> RFCN </a>
        </td>
        <td>VOCtest</td>
	<td></td>
        <td>69.93%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>16.52(bs1)</td>
        <td></td>
        <td>1 x 3 x 1344 X 1344</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SFA3D_for_Pytorch">  SFA3D_for_Pytorch </a>
        </td>
        <td>KITTI</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>0.603</td>
	<td>426(bs4)</td>
        <td></td>
        <td>bs x 3 x 608 X 608</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/SSD-MobileNetV1"> SSD-MobileNetV1 </a>
        </td>
        <td>VOC2007</td>
	<td>69.3%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>3176(bs4)</td>
        <td></td>
        <td>bs x 3 x 300 x 300</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/SSD-MobileNetV2">  SSD-MobileNetV2 </a>
        </td>
        <td>VOC2007</td>
	<td>69.8%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>2923(bs4)</td>
        <td></td>
        <td>bs x 3 x 300 x 300</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/SSD_resnet34">  SSD_resnet34 </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>20%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>37(bs1)</td>
        <td></td>
        <td>bs x 3 x 1200 x 1200</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/swin_transformer">   swin_transformer </a>
        </td>
        <td>coco</td>
	<td>47.9%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>5.9(bs8)</td>
        <td></td>
        <td>bs x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/TextSnake">   TextSnake </a>
        </td>
        <td>TextSnake</td>
	<td></td>
        <td></td>
        <td></td>
        <td>59%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>180.36(bs1)</td>
        <td></td>
        <td>1 x 3 x 512 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/TOOD">   TOOD </a>
        </td>
        <td>coco</td>
	<td>42.2%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>14.6(bs1)</td>
        <td></td>
        <td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/VGG16_SSD_for_PyTorch">   VGG16_SSD_for_PyTorch </a>
        </td>
        <td>VOC</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>77.26%</td>
        <td></td>
        <td></td>
	<td>751(bs16)</td>
        <td></td>
        <td>bs x 3 x 300 x 300</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/YOLOF">  YOLOF </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>42.8%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>126(bs16)</td>
        <td></td>
        <td>bs x 3 x 608 x 608</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/yolor">   yolor </a>
        </td>
        <td>coco</td>
	<td>52.1%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>40.9(bs8)</td>
        <td></td>
        <td>bs x 3 x 1344 x 1344</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/YoloX_Tiny_for_Pytorch">   YoloX_Tiny_for_Pytorch </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>33.1%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>890(bs64)</td>
        <td></td>
        <td>bs x 3 x 640 x 640</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/YoloXs_for_Pytorch">   YoloXs_for_Pytorch </a>
        </td>
        <td>coco</td>
	<td></td>
        <td>40.1%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>890(bs4)</td>
        <td></td>
        <td>bs x 3 x 640 x 640</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/YOLOX-mmdetection"> YOLOX-mmdetection </a>
        </td>
        <td>coco</td>
	<td>51%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>77(bs64)</td>
        <td></td>
        <td>bs x 3 x 640 x 640</td>
</table>

CV-segmentation

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=7>精度</th>
	<th colspan=2>最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>Acc</td>
        <td nowrap="nowrap">Liver 1_Dice</td>
        <td>AP</td>
        <td>mAP</td>
        <td>mIOU</td>
        <td>maxF</td>
        <td>MAE</td>
        <td>310P</td>
        <td>310</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/3D_HRNet"> 3D_HRNet </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>80.83%</td>
        <td></td>
        <td></td>
	<td>9(bs1)</td>
        <td></td>
	<td>bs x 3 x 1024 x 2048</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/3D_Nested_Unet"> 3D_Nested_Unet </a>
        </td>
        <td>Task03_Liver</td>
	<td></td>
        <td>96.5%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>3.98(bs1)</td>
        <td></td>
	<td>1 x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Cascade_Mask_RCNN">  Cascade_Mask_RCNN </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td>36.29%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>18(bs1)</td>
        <td></td>
	<td>1 x 3 x 1344 x 1344</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Cascade_Mask_RCNN_UniFormer">  Cascade_Mask_RCNN_UniFormer </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td></td>
        <td>72%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>3(bs1)</td>
        <td></td>
	<td>1 x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Cascade_RCNN">   Cascade_RCNN </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td></td>
        <td>44.2%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>9(bs1)</td>
        <td></td>
	<td>1 x 3 x 1344 x 1244</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/DeeplabV3+">  DeeplabV3+ </a>
        </td>
        <td>VOCtrainval</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>78.43%</td>
        <td></td>
        <td></td>
	<td>165(bs1)</td>
        <td></td>
	<td>bs x 3 x 513 x 513</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/ENet"> ENet </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>54.11%</td>
        <td></td>
        <td></td>
	<td>1327(bs4)</td>
        <td></td>
	<td>bs x 3 x 480x 480</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/ErfNet">  ErfNet </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>72.2%</td>
        <td></td>
        <td></td>
	<td>381(bs8)</td>
        <td></td>
	<td>bs x 3 x 512 x 1024</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/FastSCNN">   FastSCNN </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>68.6%</td>
        <td></td>
        <td></td>
	<td>39(bs1)</td>
        <td></td>
	<td>bs x 3 x 1024 x 2048</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/FCN-8s"> FCN-8s </a>
        </td>
        <td>VOC2012</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>69.01%</td>
        <td></td>
        <td></td>
	<td>84(bs1)</td>
        <td></td>
 	<td>1 x 3 x 500 x 500</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/GCNet">  GCNet </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td>61%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>13(bs1)</td>
        <td></td>
	<td>1 x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/ICNet">  ICNet </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>68.9%</td>
        <td></td>
        <td></td>
	<td>32(bs8)</td>
        <td></td>
	<td>bs x 3 x 1024 x 2048</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/IntraDA">  IntraDA </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>47.01%</td>
        <td></td>
        <td></td>
	<td>47(bs1)</td>
        <td></td>
	<td>bs x 3 x 512 x 1024</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/LPRNet_for_PyTorch">  LPRNet_for_PyTorch </a>
        </td>
        <td>代码仓提供</td>
	<td>90.2%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>27313(bs32)</td>
        <td></td>
	<td>bs x 3 x 24 x 94</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Maskrcnn-mmdet">   Maskrcnn-mmdet </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td></td>
        <td>59%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>11(bs1)</td>
        <td></td>
	<td>1 x 3 x 1216 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Nested_UNet">  Nested_UNet </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>83.8%</td>
        <td></td>
        <td></td>
	<td>2623(bs4)</td>
        <td></td>
	<td>bs x 3 x 96 x 96</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/OCRNet"> OCRNet </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>79.63%</td>
        <td></td>
        <td></td>
	<td>13(bs1)</td>
        <td></td>
	<td>bs x 3 x 1024 x 1024</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/PointRend"> PointRend </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>78.85%</td>
        <td></td>
        <td></td>
	<td>1.27(bs1)</td>
        <td></td>
	<td>1 x 3 x 1024 x 2048</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/PraNet">  PraNet </a>
        </td>
        <td>kvasir</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>83.6%</td>
        <td></td>
        <td></td>
	<td>425(bs4)</td>
        <td></td>
	<td>bs x 3 x 352 x 352</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/PSPNet">  PSPNet </a>
        </td>
        <td>VOC2012</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>76.18%</td>
         <td></td>
        <td></td>
	<td>67.8(bs16)</td>
        <td></td>
	<td>bs x 3 x 500x 500</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/RefineNet">   RefineNet </a>
        </td>
        <td>VOC2012</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>78.6%</td>
        <td></td>
        <td></td>
	<td>87(bs1)</td>
        <td></td>
	<td>bs x 3 x 500 x 500</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Segformer"> Segformer </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>75.94%</td>
        <td></td>
        <td></td>
	<td>10.65(bs4)</td>
        <td></td>
	<td>bs x 3 x 1024 x 2048</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Segmenter"> Segmenter </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>78.89%</td>
        <td></td>
        <td></td>
	<td>3.4(bs1)</td>
        <td></td>
	<td>bs x 3 x 768 x 768</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/SeMask">  SeMask </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>76.54%</td>
        <td></td>
        <td></td>
	<td>4.7(bs1)</td>
        <td></td>
	<td>bs x 3 x 1024 x 2048</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/SETR">   SETR </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>77.35%</td>
        <td></td>
        <td></td>
	<td>3.4(bs1)</td>
        <td></td>
	<td>1 x 3 x 768 x 768</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/SiamMask"> SiamMask </a>
        </td>
        <td>VOT2016</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>42.7%</td>
        <td></td>
        <td></td>
	<td></td>
        <td>302(bs1)</td>
	<td>多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/SOLOV1"> SOLOV1 </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td>32.1%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>10(bs1)</td>
        <td></td>
	<td>1 x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/STDC"> STDC </a>
        </td>
        <td>Cityscapes</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>71.81%</td>
        <td></td>
        <td></td>
	<td>27(bs1)</td>
        <td></td>
	<td>1 x 3 x 1024 x 2048</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/swin_97">  swin_97 </a>
        </td>
        <td>ADE20K</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>44.78%</td>
        <td></td>
        <td></td>
	<td>21(bs1)</td>
        <td></td>
	<td>bs x 3 x 512 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/swin_98">  swin_98 </a>
        </td>
        <td>ADE20K</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>47.92%</td>
        <td></td>
        <td></td>
	<td>16(bs1)</td>
        <td></td>
	<td>bs x 3 x 512 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/swin_99">  swin_99 </a>
        </td>
        <td>ADE20K</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>48.29%</td>
        <td></td>
        <td></td>
	<td>14(bs1)</td>
        <td></td>
	<td>bs x 3 x 512 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/swin100"> swin100 </a>
        </td>
        <td>ADE20K</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>48.74%</td>
        <td></td>
        <td></td>
	<td>14(bs1)</td>
        <td></td>
	<td>bs x 3 x 512 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Swin-Transformer-Semantic-Segmentation"> Swin-Transformer-Semantic-Segmentation </a>
        </td>
        <td>ADE20K</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>48.06%</td>
        <td></td>
        <td></td>
	<td>19(bs1)</td>
        <td></td>
	<td>1 x 3 x 512 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Transformer-SSL"> Transformer-SSL </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td>68.8%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>4(bs1)</td>
        <td></td>
	<td>1 x 3 x 800 x 1216</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/U2-Net_for_PyTorch"> U2-Net_for_PyTorch </a>
        </td>
        <td>ECSSD</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>94.8%</td>
        <td>0.033</td>
	<td>240(bs1)</td>
        <td></td>
	<td>bs x 3 x 320 x 320</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Ultra-Fast-Lane-Detection"> Ultra-Fast-Lane-Detection </a>
        </td>
        <td>Tusimple</td>
	<td>95.8%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>2254(bs32)</td>
        <td></td>
	<td>bs x 3 x 288 x 800</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/VNet"> VNet </a>
        </td>
        <td>LUNA16</td>
	<td>99.4%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>44(bs1)</td>
        <td></td>
	<td>bs x 64 x 80 x 80</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/Wseg">  Wseg </a>
        </td>
        <td>VOC</td>
	<td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>62.7%</td>
        <td></td>
        <td></td>
	<td>5(bs1)</td>
        <td></td>
	<td nowrap="nowrap">bs x 3 x 1020 x 1020</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/YOLACT">   YOLACT </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td></td>
        <td>32.07%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>163(bs1)</td>
        <td></td>
	<td>bs x 3 x 550 x 550</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/YolactEdge_for_PyTorch">   YolactEdge_for_PyTorch </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td></td>
        <td>27.96%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>270(bs1)</td>
        <td></td>
	<td>bs x 3 x 550 x 550</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/YOLACT_plus">   YOLACT_plus </a>
        </td>
        <td>coco</td>
	<td></td>
        <td></td>
        <td></td>
        <td>34.9%</td>
        <td></td>
        <td></td>
        <td></td>
	<td></td>
        <td>31(bs8)</td>
	<td>bs x 3 x 550 x 550</td>
</table>

CV-face

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=2>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>Acc</td>
        <td nowrap="nowrap">mAP</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/face/AlignedReID">  AlignedReID </a>
        </td>
        <td>Market1501</td>
	<td>80.55%</td>
        <td></td>
	<td>5293(bs32)</td>
	<td>bs x 3 x 256 x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/face/centroids-reid"> centroids-reid </a>
        </td>
        <td>DukeMTMC-reID</td>
	<td>96.8%</td>
        <td></td>
	<td>4287(bs8)</td>
	<td>bs x 3 x 256 x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/face/FaceBoxes"> FaceBoxes </a>
        </td>
        <td>FDDB</td>
	<td>94.8%</td>
        <td></td>
	<td>2332(bs1)</td>
	<td>bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/face/FaceNet">  FaceNet </a>
        </td>
        <td>LFW</td>
	<td>99.2%</td>
        <td></td>
	<td>7964(bs16)</td>
	<td>bs x 3 x 160 x 160</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/face/reid_PCB_baseline">   reid_PCB_baseline </a>
        </td>
        <td>Market</td>
	<td>92.1%</td>
        <td></td>
	<td>2031(bs16)</td>
	<td>bs x 3 x 384 x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/face/ReId-MGN-master">   ReId-MGN-master </a>
        </td>
        <td>Market</td>
	<td></td>
        <td>94.23%</td>
	<td>1519(bs8)</td>
	<td>bs x 3 x 384 x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/face/Retinaface"> Retinaface </a>
        </td>
        <td>WiderFace</td>
	<td>87.56%</td>
        <td></td>
	<td>1502(bs16)</td>
	<td>bs x 3 x 1000 x 1000</td>
</table>

CV-gan

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td nowrap="nowrap">IS</td>
        <td>FID</td>
        <td>Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/BigGAN"> BigGAN </a>
        </td>
        <td>噪声数据</td>
        <td>94.009</td>
        <td>10</td>
        <td></td>
	<td>544(bs16)</td>
	<td nowrap="nowrap">bs x 1 x 20 <br> bs x 5 x 148</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/CGAN"> CGAN </a>
        </td>
        <td>随机数</td>
        <td></td>
        <td></td>
        <td></td>
	<td>1935(bs1)</td>
	<td nowrap="nowrap">1 x 100 x 72</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/CycleGAN"> CycleGAN </a>
        </td>
        <td>maps</td>
        <td></td>
        <td></td>
        <td>CycleGAN_Ga 1 <br> CycleGAN_Gb 0.99</td>
	<td nowrap="nowrap">CycleGAN_Ga 231(bs64) <br> CycleGAN_Gb 232(bs64)</td>
	<td nowrap="nowrap">bs x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/DCGAN"> DCGAN </a>
        </td>
        <td>噪声数据</td>
        <td></td>
        <td></td>
        <td>1</td>
	<td nowrap="nowrap">108781(bs32)</td>
 	<td nowrap="nowrap">bs x 100 x 1 x 1</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/DG-Net"> DG-Net </a>
        </td>
        <td>Market-1501</td>
        <td></td>
        <td>18.12</td>
        <td></td>
	<td nowrap="nowrap">584(bs8)</td>
 	<td nowrap="nowrap">bs x 1 x 256 x 128 <br> bs x 3 x 256 x 128</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/GAN"> GAN </a>
        </td>
        <td>随机数</td>
        <td></td>
        <td></td>
        <td></td>
	<td nowrap="nowrap">496239(bs64)</td>
	<td nowrap="nowrap">bs x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/Pix2Pix"> Pix2Pix </a>
        </td>
        <td>facades</td>
        <td></td>
        <td></td>
        <td></td>
	<td nowrap="nowrap">963(bs32)</td>
	<td nowrap="nowrap">bs x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/gan/Pix2pixHD">  Pix2pixHD </a>
        </td>
        <td>cityscapes</td>
        <td></td>
        <td></td>
        <td></td>
	<td nowrap="nowrap">5(bs1)</td>
	<td nowrap="nowrap">bs x 36 x 1024 x 2048</td>
</table>

CV-image_process

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=1>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>PSNR</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/image_process/Cross-Scale-Non-Local-Attention"> Cross-Scale-Non-Local-Attention </a>
        </td>
        <td>Set5</td>
        <td>32.57</td>
	<td>0.71(bs1)</td>
	<td nowrap="nowrap">1 x 3 x 56 x 56</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/image_process/DnCNN"> DnCNN </a>
        </td>
        <td>dncnn</td>
        <td>31.53</td>
	<td>166(bs16)</td>
	<td nowrap="nowrap">bs x 1 x 481 x 481</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/image_process/SRFlow">  SRFlow </a>
        </td>
        <td>DIV2K</td>
        <td>23</td>
	<td>0.7(bs1)</td>
	<td nowrap="nowrap">1 x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/image_process/wdsr">  wdsr </a>
        </td>
        <td>DIV2K</td>
        <td>34.75</td>
	<td>13(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 1020 x 1020</td>
</table>

CV-image_registration

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=1>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/image_registration/superpoint"> superpoint </a>
        </td>
        <td>HPatches</td>
        <td>80.6%</td>
	<td>2528(bs8)</td>
	<td nowrap="nowrap">bs x 1 x 240 x 320</td>
</table>

CV-image_retrieval

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=1>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>mAP</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/image_retrieval/BLIP"> BLIP </a>
        </td>
        <td>coco</td>
        <td>81.3%</td>
	<td>text:1662(bs64) <br> image:72(bs1) <br> image_feat:73(bs1)</td>
	<td nowrap="nowrap">text:bs x 35 <br> image:bs x 3 x 384 x 384 <br> image_feat:bs x 3 x 384 x 384</td>
</table>

CV-pose_estimation

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=4>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>Top1Acc</td>
        <td>Top5Acc</td>
        <td>AP</td>
        <td>MPJPE</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/3DMPPE-ROOTNET"> 3DMPPE-ROOTNET </a>
        </td>
        <td>MuPoTS</td>
        <td>31.81%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>1565(bs4)</td>
	<td nowrap="nowrap">bs x 3 x 224 x 224 <br> bs x 1</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/DEKR"> DEKR </a>
        </td>
        <td>coco</td>
        <td></td>
        <td></td>
        <td>67.7%</td>
        <td></td>
	<td>7.72(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/HigherHRNet"> HigherHRNet </a>
        </td>
        <td>coco</td>
        <td></td>
        <td></td>
        <td>67.1%</td>
        <td></td>
	<td>185(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/HRNet_mmlab_for_pytorch"> HRNet_mmlab_for_pytorch </a>
        </td>
        <td>coco</td>
        <td></td>
        <td></td>
        <td>65.3%</td>
        <td></td>
	<td>151(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/MSPN"> MSPN </a>
        </td>
        <td>coco</td>
        <td></td>
        <td></td>
        <td>74.1%</td>
        <td></td>
	<td>933(bs4)</td>
	<td nowrap="nowrap">bs x 3 x 256 x 192</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/PoseC3D">  PoseC3D </a>
        </td>
        <td>HMDB51</td>
        <td>69.22%</td>
        <td>91.31%</td>
        <td></td>
        <td></td>
	<td>22.3(bs8)</td>
	<td nowrap="nowrap">bs x 20 x 17 x 48 x 56 x 56</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/TransPose">   TransPose </a>
        </td>
        <td>coco</td>
        <td></td>
        <td></td>
        <td>73.7%</td>
        <td></td>
	<td>500(bs4)</td>
	<td nowrap="nowrap">bs x 3 x 256 x 192</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/UniFormer">   UniFormer </a>
        </td>
        <td>coco</td>
        <td></td>
        <td></td>
        <td>93.5%</td>
        <td></td>
	<td>295(bs8)</td>
	<td nowrap="nowrap">bs x 3 x 256 x 192</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/VideoPose3D">  VideoPose3D </a>
        </td>
        <td>Human3.6M</td>
        <td></td>
        <td></td>
        <td></td>
        <td>46.6</td>
	<td>280257(bs2)</td>
	<td nowrap="nowrap">2 x 6115 x 17 x 2</td>
</table>

CV-quality_enhancement

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=3>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>ACC</td>
        <td>PSNR</td>
        <td>SSIM</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/quality_enhancement/ADNet">  ADNet </a>
        </td>
        <td>BSD68</td>
        <td>29.24%</td>
        <td></td>
        <td></td>
	<td>215(bs64)</td>
	<td nowrap="nowrap">bs x 1 x 321 x 481</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/quality_enhancement/SRGAN">   SRGAN </a>
        </td>
        <td>Set5</td>
        <td></td>
        <td>33.4391</td>
        <td>93.08%</td>
	<td>380(bs8)</td>
	<td nowrap="nowrap">bs x 3 x 140 x 140</td>
</table>

CV-super_resolution

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=2>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>PSNR</td>
        <td>SSIM</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/FLAVR_for_PyTorch">  FLAVR_for_PyTorch </a>
        </td>
        <td>UCF101</td>
        <td>29.83</td>
        <td>94.46%</td>
	<td>77(bs16)</td>
	<td nowrap="nowrap">bs x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/super_resolution/RCAN">  RCAN </a>
        </td>
        <td>Set5</td>
        <td>38.25</td>
        <td>96.06%</td>
	<td>12(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/super_resolution/RDN">   RDN </a>
        </td>
        <td>Set5</td>
        <td>38.27</td>
        <td></td>
	<td>47(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 114 x 114</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/super_resolution/Real-ESRGAN">  Real-ESRGAN </a>
        </td>
        <td>代码仓提供</td>
        <td></td>
        <td></td>
	<td>251(bs4)</td>
	<td nowrap="nowrap">bs x 3 x 220 x 220 <br> bs x 3 x 64 x 64</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/super_resolution/SRCNN">  SRCNN </a>
        </td>
        <td>Set5</td>
        <td>36.33</td>
        <td></td>
	<td>2361(bs1)</td>
	<td nowrap="nowrap">bs x 1 x 256 x 256</td>
</table>

CV-tracking

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=5>精度</th>
	<th colspan=2>最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>success_score</td>
        <td>precision_score</td>
        <td>Acc</td>
        <td>EPE</td>
        <td>MAPE</td>
        <td>310P</td>
        <td>310</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Flownet2_for_Pytorch"> Flownet2_for_Pytorch </a>
        </td>
        <td>MPI-Sintel-complete</td>
        <td></td>
        <td></td>
        <td></td>
        <td>2.15</td>
        <td></td>
	<td>14(bs1)</td>
        <td></td>
	<td nowrap="nowrap">bs x 3 x 448 x 1024</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/tracking/SiamFC">  SiamFC </a>
        </td>
        <td>OTB2015</td>
        <td>57.2%</td>
        <td>76.2%</td>
        <td></td>
        <td></td>
        <td></td>
	<td>exemplar_bs1:6072(bs1) <br> search_bs1:948(bs1)</td>
        <td></td>
	<td nowrap="nowrap">1 x 3 x 255 x 255 <br> 1 x 9 x 127 x 127</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/tracking/SiamRPN">  SiamRPN </a>
        </td>
        <td>VOT</td>
        <td></td>
        <td></td>
        <td>63.9%</td>
        <td></td>
        <td></td>
	<td></td>
        <td>42(bs1)</td>
	<td nowrap="nowrap">1 x 3 x 127 x 127 <br> 1 x 3 x 255 x 255</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/tracking/GMA">  GMA </a>
        </td>
        <td>MPI-Sintel-complete</td>
        <td></td>
        <td></td>
        <td>final:88.95% <br> clean:92.65%</td>
        <td></td>
        <td></td>
	<td>0.77(bs4)</td>
        <td></td>
	<td nowrap="nowrap">1 x 3 x 440 x 1024</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/deepctr">  deepctr </a>
        </td>
        <td>代码仓提供</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>WDL:2.147 <br> xDeepFM:1.97 <br> AutoInt:2.14</td>
	<td>WDL:0.079(bs40) <br> xDeepFM:0.177(bs40) <br> AutoInt:0.22(bs40)</td>
        <td></td>
	<td nowrap="nowrap">40 x 6</td>
</table>

CV-video_understanding

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=4>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>ADK</td>
        <td>MKR</td>
        <td>Top1Acc</td>
        <td>Top5Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/video_understanding/FOMM">  FOMM </a>
        </td>
        <td>taichi</td>
        <td>6.7975</td>
        <td>0.036</td>
        <td></td>
        <td></td>
	<td>kp detector:957(bs1) <br> generator:7(bs1)</td>
	<td nowrap="nowrap">kp detecto:1 x 3 x 256 x 256 <br> generator:多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/video_understanding/I3D_Nonlocal">  I3D_Nonlocal </a>
        </td>
        <td>kinetics400</td>
        <td></td>
        <td></td>
        <td>70.07%</td>
        <td></td>
	<td>14(bs1)</td>
	<td nowrap="nowrap">bs x 10 x 3 x 32 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/video_understanding/NonLocal">   NonLocal </a>
        </td>
        <td>kinetics400</td>
        <td></td>
        <td></td>
        <td>71.62%</td>
        <td>90.27%</td>
	<td>97(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 8 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/video_understanding/SlowFast">   SlowFast </a>
        </td>
        <td>kinetics400</td>
        <td></td>
        <td></td>
        <td>70.07%</td>
        <td>88.55%</td>
	<td>138(bs1)</td>
	<td nowrap="nowrap">bs x 3 x 32 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/video_understanding/TSM">  TSM </a>
        </td>
        <td>UCF-101</td>
        <td></td>
        <td></td>
        <td>94.48%</td>
        <td>99.63%</td>
	<td>194(bs1)</td>
	<td nowrap="nowrap">bs x 8 x 3 x 224 x 224</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/video_understanding/TSN">  TSN </a>
        </td>
        <td>UCF-101</td>
        <td></td>
        <td></td>
        <td>82.83%</td>
        <td></td>
	<td>22.19(bs32)</td>
	<td nowrap="nowrap">bs x 75 x 3 x 256 x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/video_understanding/X3D"> X3D </a>
        </td>
        <td>Kinetic400</td>
        <td></td>
        <td></td>
        <td>73.75%</td>
        <td>90.25%</td>
	<td>386(bs8)</td>
	<td nowrap="nowrap">bs x 3 x 13 x 182 x 182</td>
</table>

Audio

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=4>精度</th>
	<th colspan=2>最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>EER</td>
        <td>WER</td>
        <td>ROC_AUC</td>
        <td>mel_loss</td>
        <td>310P</td>
        <td>310</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/AASIST-L_for_Pytorch">  AASIST-L_for_Pytorch </a>
        </td>
        <td>LA</td>
        <td>0.979</td>
        <td></td>
        <td></td>
        <td></td>
	<td>168(bs64)</td>
        <td></td>
	<td nowrap="nowrap">1 x 64600</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/data2vec">  data2vec </a>
        </td>
        <td>LibriSpeech</td>
        <td></td>
        <td>0.94</td>
        <td></td>
        <td></td>
	<td>11(bs1)</td>
        <td></td>
	<td nowrap="nowrap">bs x 559280</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/Ecapa_Tdnn">  Ecapa_Tdnn </a>
        </td>
        <td>VoxCeleb1</td>
        <td></td>
        <td></td>
        <td>0.9991</td>
        <td></td>
	<td>1536(bs4)</td>
        <td></td>
	<td nowrap="nowrap">bs x 80 x 200</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/FastPitch"> FastPitch </a>
        </td>
        <td>LJSpeech</td>
        <td></td>
        <td></td>
        <td></td>
        <td>11.33</td>
	<td>126(bs8)</td>
        <td></td>
 	<td nowrap="nowrap">bs x 200</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/jasper">  jasper </a>
        </td>
        <td>LibriSpeech</td>
        <td></td>
        <td>9.709</td>
        <td></td>
        <td></td>
	<td>41(bs1)</td>
        <td></td>
	<td nowrap="nowrap">bs x 64 x 4000</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/LSTM">  LSTM </a>
        </td>
        <td>timit</td>
        <td></td>
        <td>18.9075</td>
        <td></td>
        <td></td>
	<td>83.4(bs64)</td>
        <td></td>
	<td nowrap="nowrap">bs x 390 x 243</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/rawnet2">  rawnet2 </a>
        </td>
        <td>VoxCeleb1</td>
        <td>2.5%</td>
        <td></td>
        <td></td>
        <td></td>
	<td></td>
        <td>77(bs16)</td>
	<td nowrap="nowrap">bs x 59049</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/Speech-Transformer">  Speech-Transformer </a>
        </td>
        <td>aishell</td>
        <td>9.9%</td>
        <td></td>
        <td></td>
        <td></td>
	<td></td>
        <td>0.82(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/tdnn">  tdnn </a>
        </td>
        <td>librispeech</td>
        <td></td>
        <td>98.69%</td>
        <td></td>
        <td></td>
	<td></td>
        <td>1562(bs16)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/wav2vec2"> wav2vec2 </a>
        </td>
        <td>librispeech</td>
        <td></td>
        <td>2.96%</td>
        <td></td>
        <td></td>
	<td>157(bs16)</td>
        <td></td>
	<td nowrap="nowrap">bs x 10000</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/audio/WaveGlow"> WaveGlow </a>
        </td>
        <td>LJSpeech</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>3(1 x 80 x 154)</td>
        <td></td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/Wenet2_for_Pytorch"> WeNet </a>
        </td>
        <td>aishell</td>
        <td></td>
        <td>4.68%</td>
        <td></td>
        <td></td>
	<td>150.28(bs1)</td>
        <td></td>
	<td nowrap="nowrap">多尺度</td>
</table>

Knowledge

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=1>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>MRR</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/knowledge/RotatE">   RotatE </a>
        </td>
        <td>FB15k-237</td>
        <td>0.3355</td>
	<td>head:222(bs64) <br> tail:222(bs64)</td>
	<td nowrap="nowrap">bs x 3 <br> bs x 14541</td>
</table>

Nlp 

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=6>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>Acc</td>
        <td>WER</td>
        <td>loss</td>
        <td>BLEU</td>
        <td>F1</td>
        <td>bpc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Bert_Base_Uncased_for_Pytorch"> Bert_Base_Uncased_for_Pytorch </a>
        </td>
        <td>squad</td>
        <td>88.78%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>221(bs4)</td>
	<td nowrap="nowrap">bs x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/BertSum"> BertSum </a>
        </td>
        <td>代码仓提供</td>
        <td>42.85%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>138.09(bs8)</td>
	<td nowrap="nowrap">bs x 512 <br> bs x 37</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/ch_ppocr_server_v2.0_rec"> ch_ppocr_server_v2.0_rec </a>
        </td>
        <td>PaddleOCR</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>289(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/ch_PP-OCRv2_rec">  ch_PP-OCRv2_rec </a>
        </td>
        <td>PaddleOCR</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>260(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/ch_PP-OCRv3_rec">  ch_PP-OCRv3_rec </a>
        </td>
        <td>PaddleOCR</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>1411(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/CNN_Transformer_for_Pytorch"> CNN_Transformer_for_Pytorch </a>
        </td>
        <td>Librispeech</td>
        <td></td>
        <td>0.0556</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>52(bs1)</td>
	<td nowrap="nowrap">多尺度</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/deberta"> deberta </a>
        </td>
        <td>MNLI</td>
        <td>90.46%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>5(bs64)</td>
	<td nowrap="nowrap">bs x 256</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/ELMo"> ELMo </a>
        </td>
        <td>1 Billion Word</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>62(bs1)</td>
	<td nowrap="nowrap">1 x 8 x 50</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/en_PP-OCRv3_rec">  en_PP-OCRv3_rec </a>
        </td>
        <td>PaddleOCR</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>6144(bs16)</td>
	<td nowrap="nowrap">bs x 3 x 48 x 320</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/GPT2_for_Pytorch">  GPT2_for_Pytorch </a>
        </td>
        <td>wiki_zh_2019</td>
        <td></td>
        <td></td>
        <td>16.5</td>
        <td></td>
        <td></td>
        <td></td>
	<td>189(bs16)</td>
	<td nowrap="nowrap">bs x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/GNMT">   GNMT </a>
        </td>
        <td>newstest2014</td>
        <td></td>
        <td></td>
        <td></td>
        <td>22.69</td>
        <td></td>
        <td></td>
	<td>24(bs1)</td>
	<td nowrap="nowrap">1 x 1 <br> 1 x 30</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/hubert">   hubert </a>
        </td>
        <td>test-clean</td>
        <td></td>
        <td>2.13</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>3(bs1)</td>
	<td nowrap="nowrap">1 x 580000</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/RARE_Resnet34_vd">  RARE_Resnet34_vd </a>
        </td>
        <td>LMDB</td>
        <td>84.79%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>1603(bs32)</td>
	<td nowrap="nowrap">bs x 3 x 32 x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/Rosetta_MobileNetV3"> Rosetta_MobileNetV3 </a>
        </td>
        <td>LMDB</td>
        <td>77.38%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>24219(bs64)</td>
	<td nowrap="nowrap">bs x 3 x 32 x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/Rosetta_Resnet34_vd"> Rosetta_Resnet34_vd </a>
        </td>
        <td>LMDB</td>
        <td>80.63%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>7418(bs16)</td>
	<td nowrap="nowrap">bs x 3 x 32 x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/SATRN">  SATRN </a>
        </td>
        <td>IIIT5K</td>
        <td>94.87%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>31(bs1)</td>
	<td nowrap="nowrap">1 x 3 x 32 x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/SpanBERT">  SpanBERT </a>
        </td>
        <td>SQuAD 1.1</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>93.95%</td>
        <td></td>
	<td>43(bs1)</td>
	<td nowrap="nowrap">bs x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/StarNet_MobileNetV3">  StarNet_MobileNetV3 </a>
        </td>
        <td>LMDB</td>
        <td>80.02%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>2617(bs64)</td>
	<td nowrap="nowrap">bs x 3 x 32 x 100</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/TextCNN">   TextCNN </a>
        </td>
        <td>THUCNews</td>
        <td>90.47%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>29237(bs64)</td>
	<td nowrap="nowrap">bs x 32</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/tinybert">   tinybert </a>
        </td>
        <td>SST-2</td>
        <td>92.32%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>11160(bs64)</td>
	<td nowrap="nowrap">bs x 4 x 84 x 84</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/Transformer">    Transformer </a>
        </td>
        <td>Multi30k</td>
        <td>40.92%</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>48(bs1)</td>
	<td nowrap="nowrap">1 x 15</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/TransformerXL_for_Pytorch">   TransformerXL_for_Pytorch </a>
        </td>
        <td>enwik8</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>1.966</td>
	<td>287(bs1)</td>
	<td nowrap="nowrap">80 x 1 <br> 160 x 1 x 512</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/TrOCR">  TrOCR </a>
        </td>
        <td>IAM</td>
        <td></td>
        <td>4.25</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
	<td>8(bs1)</td>
	<td nowrap="nowrap">1 x 3 x 384 x 384</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/VilBert_for_Pytorch"> VilBert_for_Pytorch </a>
        </td>
        <td>coco</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>0.67</td>
        <td></td>
	<td>493(bs32)</td>
	<td nowrap="nowrap">多尺度</td>
</table>

RL

<table align="center">
    <tr>
        <th rowspan=2>Name</th>
	<th rowspan=2>Dataset</th>
        <th align="center" colspan=1>精度</th>
	<th rowspan=2>310P最优性能（对应bs）</th>
	<th rowspan=2>输入shape</th>
    </tr>
    <tr>
        <td>Acc</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/rl/c51">  c51 </a>
        </td>
        <td>随机数</td>
        <td>98.9%</td>
	<td>6050(bs1)</td>
	<td nowrap="nowrap">1 x 4 x 84 x 84</td>
    </tr>
    <tr>
        <td>
        <a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/rl/DQN"> DQN </a>
        </td>
        <td>随机数</td>
        <td>100%</td>
	<td>8147(bs1)</td>
	<td nowrap="nowrap">1 x 4 x 84 x 84</td>
</table>
