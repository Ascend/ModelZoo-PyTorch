SCRIPT_DIR=`pwd`
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')

cp $TRANSFORMER_PACKAGE_PATH/models/bloom/modeling_bloom.py $TRANSFORMER_PACKAGE_PATH/models/bloom/modeling_bloom.py.bak
cp $SCRIPT_DIR/patches/utils.patch $TRANSFORMER_PACKAGE_PATH/generation/
cd $TRANSFORMER_PACKAGE_PATH/generation/
cp utils.py utils.py.bak
patch -p1 < utils.patch
cd $SCRIPT_DIR

case "${RUN_OPTION}" in
    "--run")
        cp $SCRIPT_DIR/patches/modeling.patch $TRANSFORMER_PACKAGE_PATH/models/bloom/
        cd $TRANSFORMER_PACKAGE_PATH/models/bloom/
        patch -p1 < modeling.patch
        cd $SCRIPT_DIR
        export TASK_QUEUE_ENABLE=1
        python3 run_bloom_npu.py
        ;;
    "--parallel")
        bash $SCRIPT_DIR/cut_model_and_run_bloom.sh
        ;;
    *)
        echo "unknown build type:${RUN_OPTION}"
        echo "run.sh [--run|--parallel]"
        ;;
esac

if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_bloom.py.bak" ];then
    mv $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py.bak $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py
fi

if [ -f "$TRANSFORMER_PACKAGE_PATH/generation/utils.py.bak" ];then
    mv $TRANSFORMER_PACKAGE_PATH/generation/utils.py.bak $TRANSFORMER_PACKAGE_PATH/generation/utils.py
fi