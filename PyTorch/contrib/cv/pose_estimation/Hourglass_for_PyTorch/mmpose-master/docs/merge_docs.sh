#!/usr/bin/env bash

sed -i '$a\\n' ../configs/bottom_up/*/*.md
sed -i '$a\\n' ../configs/top_down/*/*.md
sed -i '$a\\n' ../demo/*_demo.md
sed -i '$a\\n' ../configs/hand/*/*.md

cat  ../configs/bottom_up/*/*.md > bottom_up_models.md
cat  ../configs/top_down/*/*.md > top_down_models.md
cat  ../demo/*_demo.md > demo.md
cat  ../configs/hand/*/*.md > hand_models.md

sed -i "s/#/#&/" bottom_up_models.md
sed -i "s/#/#&/" top_down_models.md
sed -i "s/#/#&/" demo.md
sed -i "s/#/#&/" hand_models.md
sed -i "s/md###t/html#t/g" bottom_up_models.md
sed -i "s/md###t/html#t/g" top_down_models.md
sed -i "s/md###t/html#t/g" demo.md
sed -i "s/md###t/html#t/g" hand_models.md

sed -i '1i\# Bottom Up Models' bottom_up_models.md
sed -i '1i\# Top Down Models' top_down_models.md
sed -i '1i\# Demo' demo.md
sed -i '1i\# Hand Models' hand_models.md

sed -i 's/](\/docs\//](/g' bottom_up_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' top_down_models.md
sed -i 's/](\/docs\//](/g' hand_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' bottom_up_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' top_down_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' hand_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' getting_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' benchmark.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' config.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' changelog.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' demo.md
sed -i 's/](\/docs\//](/g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tutorials/*.md
sed -i 's/](\/docs\//](/g' data_preparation.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' data_preparation.md
