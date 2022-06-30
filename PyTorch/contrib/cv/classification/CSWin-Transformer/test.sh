FPS=`grep Train test/output/0/train_0.log|grep -a '100/625'|awk -F "100/625" '{print $NF}' |awk -F " " '{print substr($7,0,length($7)-2)}'#|awk 'END {print}'`
echo $FPS
