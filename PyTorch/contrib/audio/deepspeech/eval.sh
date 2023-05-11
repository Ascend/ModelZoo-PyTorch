source scripts/env_new.sh
python3 -u test.py \
        model.model_path=models/deepspeech_final.pth \
        test_manifest=data/an4_test_manifest.csv

