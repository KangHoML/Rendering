#!/bin/bash

DATA_ROOT=data/Replica
OUTPUT_ROOT=output/Replica

# Replica 하위 디렉토리
for dir in "$DATA_ROOT"/*; do
    if [ -d "$dir" ]; then
        scene_name=$(basename "$dir")
        echo "Processing scene: $scene_name"

        # 각 장면에 대해 임베딩
        cd encoders/lseg_encoder
        python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv \
               --outdir ../../"$dir"/rgb_feature_langseg --test-rgb-dir ../../"$dir"/images --workers 0
        
        # 각 장면에 대해 학습
        cd ../../
        python train.py -s "$dir" -m $OUTPUT_ROOT/"$scene_name" -f lseg --speedup --iterations 7000
    
    fi
done    