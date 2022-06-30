output_root="./output_bs$1"

rm -rf ${output_root}
mkdir ${output_root}
cd ./result/*
num2=0
for file_old in `ls | grep output_0`
do
        num2=`expr $num2 + 1`
        file_new="../../output_bs$1/${file_old}"
        cp ${file_old} ${file_new}
done
cd ../..
rm -rf ./result
echo result_nums:${num2}