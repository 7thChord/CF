#!/bin/bash

export VBENCH_CACHE_DIR=/shared/bofeng/code/VBench/pretrained

cd /shared/bofeng/code/VBench

VIDEOS_PATHS=(
    "/shared/bofeng/code/inconsistent_videos/small_artifacts"
    "/shared/bofeng/code/inconsistent_videos/large_artifacts"
)

MASTER_PORT=29500
for VIDEOS_PATH in "${VIDEOS_PATHS[@]}"; do
    OUTPUT_PATH="$VIDEOS_PATH/evaluation_results"
    echo "Evaluating videos in: $VIDEOS_PATH"

    MASTER_PORT=$MASTER_PORT python3 evaluate.py \
        --videos_path "$VIDEOS_PATH" \
        --output_path "$OUTPUT_PATH" \
        --dimension temporal_flickering_ratio \
        --mode custom_input &
    MASTER_PORT=$((MASTER_PORT + 1))
done

wait


