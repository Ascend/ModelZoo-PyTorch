#! /bin/bash
set -e

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

if [ ! -d "./data/dev" ]; then
	mkdir -p ./data/dev
	wget -nc -O  ./data/dev/dev.tgz http://data.statmt.org/wmt16/translation-task/dev.tgz
	tar -xvzf "./data/dev/dev.tgz" -C "./data/" dev/newstest2014-deen-src.de.sgm dev/newstest2014-deen-ref.en.sgm
fi

if [ ! -d "./mosesdecoder" ]; then
	git clone https://github.com/moses-smt/mosesdecoder.git "./mosesdecoder"
	cd ./mosesdecoder
	git reset --hard 8c5eaa1a122236bbf927bde4ec610906fea599e6
	cd ../
fi

./mosesdecoder/scripts/ems/support/input-from-sgm.perl \
	  < ./data/dev/newstest2014-deen-src.de.sgm \
	    > ./data/newstest2014.de
./mosesdecoder/scripts/ems/support/input-from-sgm.perl \
	  < ./data/dev/newstest2014-deen-ref.en.sgm \
	    > ./data/newstest2014.en
