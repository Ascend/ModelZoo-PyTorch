i=280
currentDir=`pwd`
echo "eval log path is ${currentDir}/eval.log"
{
while(( $i<=350 ))
do
        if [ $i -lt 315 ]
        then
                echo $i
                cp eval_$i/FDDB_dets.txt FDDB_Evaluation/
                cd FDDB_Evaluation
                python3 convert.py
                python3 split.py
                python3 evaluate.py
                i=`expr $i + 5`
                cd $currentDir
	elif [ $i -eq 316 ]
	then
		let i++
        else
                echo $i
                cp eval_$i/FDDB_dets.txt FDDB_Evaluation/
                cd FDDB_Evaluation
                python3 convert.py
                python3 split.py
                python3 evaluate.py
                let i++
                cd $currentDir
        fi
done
} > eval.log 2>&1 &
wait
cat eval.log|grep Average > tmp.log
python3 eval.py tmp.log | tee AP.log
echo "Final AP path is ${currentDir}/AP.log"
