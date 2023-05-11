#!/bin/bash
source scripts/npu_set_env.sh
currentDir=`pwd`
echo "test log path is ${currentDir}/test.log"
i=280
{
while(( $i<=350 ))
do
	if [ $i -lt 315 ]
	then
		echo $i
		python3 -u test.py -m weights/FaceBoxes_epoch_$i.pth --cpu --dataset FDDB --save_folder eval_$i/ &
		i=`expr $i + 5`
	elif [ $i -eq 316 ]
	then
		let i++
	else
		echo "$i"
		python3 -u test.py -m weights/FaceBoxes_epoch_$i.pth --cpu --dataset FDDB --save_folder eval_$i/ &
		let i++
	fi		
done
wait
} > test.log &
