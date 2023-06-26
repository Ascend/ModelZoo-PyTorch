cd ../
mkdir Lightning-AI
# shellcheck disable=SC2164
cd Lightning-AI
lightning_url=`sed '/^lightning_url=/!d;s/.*=//' ../../url.ini`
git clone -b 0.7.1 ${lightning_url}
mv lightning/pytorch_lightning ../
# shellcheck disable=SC2103
cd ..
rm -rf Lightning-AI
echo "pytorch_lightning_v0.7.1 has been downloaded"
cp goturn/tools/data_parallel.py pytorch_lightning/overrides/
cp goturn/tools/distrib_data_parallel.py pytorch_lightning/trainer/
cp goturn/tools/distrib_parts.py pytorch_lightning/trainer/
cp goturn/tools/evaluation_loop.py pytorch_lightning/trainer/
cp goturn/tools/lightning.py pytorch_lightning/core/
cp goturn/tools/trainer.py pytorch_lightning/trainer/

