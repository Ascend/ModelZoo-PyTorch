cd ..



echo batch_size=4

rm -rf ./result
mkdir result
./msame --model "./om/ecapa_tdnn_bs4.om" --input "./input_bs4/" --output "./result" --outfmt TXT
rm -rf ./output_bs4
mkdir output_bs4
cd ./result/*
num1=0
for file_old in `ls | grep output_0`
do
        num1=`expr $num1 + 1`
        file_new="../../output_bs4/${file_old}"
        cp ${file_old} ${file_new}
done



echo batch_size=1
cd ../..
rm -rf ./result
mkdir result
./msame --model "./om/ecapa_tdnn_bs1.om" --input "./input_bs1/" --output "./result" --outfmt TXT
rm -rf ./output_bs1
mkdir output_bs1
cd ./result/*
num2=0
for file_old in `ls | grep output_0`
do
        num2=`expr $num2 + 1`
        file_new="../../output_bs1/${file_old}"
        cp ${file_old} ${file_new}
done
echo result_nums_bs4:${num1}
echo result_nums_bs1:${num2}