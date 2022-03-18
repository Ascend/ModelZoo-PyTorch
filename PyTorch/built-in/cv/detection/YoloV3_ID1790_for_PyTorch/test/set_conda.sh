for para in $*
do
    if [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        echo "PATH CONDA BEFORE: $PATH"
        export PATH=/home/anaconda3/bin:$PATH
        export LD_LIBRARY_PATH=/home/anaconda3/lib:$LD_LIBRARY_PATH
        export PYTHONPATH=/home/anaconda3/envs/$conda_name/lib/python3.7/site-packages/mmcv:$PYTHONPATH
        echo "PATH CONDA AFTER: $PATH"
    fi
done
