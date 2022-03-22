# 删除多余的输出.bin文件
rm -rf ./result/dumpOutput_device*/*_2.bin
rm -rf ./result/dumpOutput_device*/*_3.bin
rm -rf ./result/dumpOutput_device*/*_4.bin
rm -rf ./result/dumpOutput_device*/*_5.bin

# 将其他文件夹的.bin结果移动到同一个目录下
mv ./result/dumpOutput_device1/* ./result/dumpOutput_device0/
mv ./result/dumpOutput_device2/* ./result/dumpOutput_device0/
mv ./result/dumpOutput_device3/* ./result/dumpOutput_device0/

echo 'clear2345.sh done'
