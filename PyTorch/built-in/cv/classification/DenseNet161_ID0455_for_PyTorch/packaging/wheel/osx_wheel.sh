if [[ ":$PATH:" == *"conda"* ]]; then
    echo "existing anaconda install in PATH, remove it and run script"
    exit 1
fi
# download and activate anaconda
rm -rf ~/minconda_wheel_env_tmp
miniconda3_mac_url=`sed '/^miniconda3_mac_url=/!d;s/.*=//' ../../url.ini`
wget -q ${miniconda3_mac_url} && \
    chmod +x Miniconda3-latest-MacOSX-x86_64.sh && \
    ./Miniconda3-latest-MacOSX-x86_64.sh -b -p ~/minconda_wheel_env_tmp && \
    rm Miniconda3-latest-MacOSX-x86_64.sh

. ~/minconda_wheel_env_tmp/bin/activate


export TORCHVISION_BUILD_VERSION="0.4.0.dev$(date "+%Y%m%d")"
export TORCHVISION_BUILD_NUMBER="1"
export OUT_DIR=~/torchvision_wheels

export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++

pushd /tmp
rm -rf vision
vision_url=`sed '/^vision_url=/!d;s/.*=//' ../../url.ini`
git clone ${vision_url}
pushd vision

desired_pythons=( "2.7" "3.5" "3.6" "3.7" )
# for each python
for desired_python in "${desired_pythons[@]}"
do
    # create and activate python env
    env_name="env$desired_python"
    conda create -yn $env_name python="$desired_python"
    conda activate $env_name

    pip uninstall -y torch || true
    pip uninstall -y torch_nightly || true

    export TORCHVISION_PYTORCH_DEPENDENCY_NAME=torch_nightly
    torch_nightly_cpu_url=`sed '/^torch_nightly_cpu_url=/!d;s/.*=//' ../../url.ini`
    pip install torch_nightly -f ${torch_nightly_cpu_url}
    export TORCHVISION_PYTORCH_DEPENDENCY_VERSION="$(pip show torch_nightly | grep ^Version: | sed 's/Version:  *//')"
    echo "Building against ${TORCHAUDIO_PYTORCH_DEPENDENCY_VERSION}"

    # install torchvision dependencies
    pip install ninja scipy pytest

    python setup.py clean
    python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
popd
popd
