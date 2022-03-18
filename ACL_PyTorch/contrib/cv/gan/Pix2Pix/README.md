# Pix2Pix


# 精度性能

    | 模型      | 性能基准   | 310性能   |
    | :------: | :------:  | :------:  | 
    | fsaf bs1 | 556         | 402        | 
    | fsaf bs16| 359         | 464        | 
精度直接看生成效果

    
# 自验报告
  
    # 第1次验收测试   
  	# 验收结果 OK 
  	# 验收环境: A + K / CANN 5.0.2

  
  	# pth是否能正确转换为om
  	bash ./test/pth2om.sh  --pth_path=./checkpoints/facades_label2photo_pretrained
  	# 验收结果： OK 
  	# 备注： 成功生成om，无运行报错，报错日志xx 等
  
  	# 精度数据是否达标（需要显示官网pth精度与om模型的精度）
  	# npu性能数据(确保device空闲时测试，如果模型支持多batch，测试bs1与bs16，否则只测试bs1，性能数据以单卡吞吐率为标准)
  	bash ./test/eval_acc_perf.sh --datasets_path='./datasets/facades'
  	# 验收结果： 是 
  	# 备注： 验收310测试性能bs1:402FPS bs16:464FPS；无运行报错，报错日志xx 等
  
  
  	# 310性能是否超过基准： 是 
  	bs1:310=402/556=0.723倍基准
  	bs16:310=464/359=1.292倍基准

