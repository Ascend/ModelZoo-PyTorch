source ~/anaconda3/etc/profile.d/conda.sh

## cd to root
cd ..

conda create -n mipnet python=3.8 -y
conda activate mipnet
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y ### for new gpus
pip install Cython h5py
pip install -r requirements.txt
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'## to avoid the pycoco numpy bug # https://github.com/cocodataset/cocoapi/pull/354


## install crowdpose
cd lib/crowdpose-api/crowdpose-api/PythonAPI
./install.sh
cd ../../../..

## install nms
cd lib/nms
make clean
make

cd ../..
