# SRGAN模型PyTorch离线推理指导

## 1 环境准备

1. 创建conda环境，可以clone一个已有的conda环境，并将其命名为`srgan` .

   ```shell
   conda create -n srgan --clone local_env
   ```

2. 安装必要的依赖

   ```
   pip3.7 install -r requirements.txt
   ```

3. 准备测试数据

   模型要求的数据集共包含五张图片，可以通过 [这里 (k9cj)](https://pan.baidu.com/s/1zTfNmjC5DOEMfC9gxOZc3g) 下载。

## 2 离线推理

### 2.1 310精度及性能

310上执行，执行时使用npu-smi info查看设备状态，确保device空闲，输出310相应的精度和性能

```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=./data/Challenge2_Test_Task12_Images  
```

### 2.2 基准性能

T4上执行，执行时使用nvidia-smi查看设备状态，确保device空闲

- 输出T4精度

```
bash test/eval_acc_gpu.sh
```

- 输出T4性能

```
bash test/perf_gpu.sh
```

 **评测结果：**

| 模型  | pth精度                   | 310离线推理精度          | 基准性能 | 310性能 |
| ----- | ------------------------- | ------------------------ | -------- | ------- |
| SRGAN | PSNR:33.4604  SSIM:0.9308 | PSNR:33.4401 SSIM:0.9530 | 38.6     | 190     |