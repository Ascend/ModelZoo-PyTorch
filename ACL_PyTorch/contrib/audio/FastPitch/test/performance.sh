source /usr/local/Ascend/ascend-toolkit/set_env.sh
chmod +x ./test/benchmark.x86_64

cd test

echo batch_size=1
./benchmark.x86_64 -round=20 -device_id=0 -batch_size=1 -om_path=./models/FastPitch_bs1.om

echo batch_size=4
./benchmark.x86_64 -round=20 -device_id=0 -batch_size=4 -om_path=./models/FastPitch_bs4.om

echo batch_size=8
./benchmark.x86_64 -round=20 -device_id=0 -batch_size=8 -om_path=./models/FastPitch_bs8.om

echo batch_size=16
./benchmark.x86_64 -round=20 -device_id=0 -batch_size=16 -om_path=./models/FastPitch_bs16.om

echo batch_size=32
./benchmark.x86_64 -round=20 -device_id=0 -batch_size=32 -om_path=./models/FastPitch_bs32.om