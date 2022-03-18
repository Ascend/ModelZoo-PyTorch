#for para in $*
#do
#	if [[ $para == --conda_name* ]];then
#		conda_name=`echo ${para#*=}`
#		echo "PATH CONDA BEFORE: $PATH"
#		#echo "source activate $conda_name SUCCESS"		
#		export PATH=/home/anaconda3/bin:$PATH
#		export LD_LIBRARY_PATH=/home/anaconda3/lib:$LD_LIBRARY_PATH
#		echo "PATH CONDA AFTER: $PATH"
#	fi
#done
export PATH=/home/anaconda3/bin:$PATH 
export LD_LIBRARY_PATH=/home/anaconda3/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH