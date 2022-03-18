echo "t4 bs1 perfoermance start"
python PerformanceForGPU.py --batch_size=1
echo "t4 bs16 perfoermance start"
python PerformanceForGPU.py --batch_size=16

cd gpuPerformance
tail *
