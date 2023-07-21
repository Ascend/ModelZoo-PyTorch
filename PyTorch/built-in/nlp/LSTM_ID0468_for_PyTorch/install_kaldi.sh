git config --global http.sslverify false
git clone `sed '/^kaldi_url=/!d;s/.*=//' url.ini`
cd kaldi/tools
apt update
apt install -y automake autoconf wget sox libtool subversion python2.7
extras/install_mkl.sh
extras/check_dependencies.sh
make -j 32
extras/install_irstlm.sh
extras/install_openblas.sh
cd ../src/
./configure --shared
make -j clean depend
make -j 32