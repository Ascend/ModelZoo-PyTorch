# Pix2Pix
    url=https://gitee.com/iiiimp/modelzoo/tree/master/contrib/PyTorch/Research/cv/gan/Pix2Pix
    branch=master

# 精度性能

    | 名称      | 精度      | 性能      |
    | :------: | :------:  | :------:  | 
    | GPU-1p   | -         | 15        | 
    | GPU-8p   | -         | 31        | 
    | NPU-1p   | -         | 8         | 
    | NPU-8p   | -         | 8         | 
# 自验报告
  
    # 1p train perf
    # 是否正确输出了性能log文件
    bash ./test/train_performance_1p.sh\  --data_path=./datasets/facades
    # 验收结果： OK 
    # 备注： 目标性能8FPS；验收测试性能8FPS；
    
    # 8p train perf
    # 是否正确输出了性能log文件
    bash ./test/train_performance_8p.sh\  --data_path=./datasets/facades
    # 验收结果： OK 
    # 备注： 目标性能15FPS；验收测试性能8PS；

    # 8p train full
    # 是否正确输出了性能精度log文件，是否正确保存了模型文件
    bash ./test/train_full_8p.sh\  --data_path=./datasets/facades
    # 验收结果： OK 
    # 备注：直接看图片效果

    # 8p eval
    # 是否正确输出了性能精度log文件
    bash ./test/train_eval_8p.sh\  --data_path=./datasets/facades --pth_path=./checkpoints/facades_pix2pix_npu_8p_full
    # 验收结果： OK 
    # 备注：直接看图片效果

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
