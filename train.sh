#!/bin/sh
unset http_proxy
unset https_proxy
unset all_proxy
export no_proxy="localhost,127.0.0.1"

failed_scans=""

for i in 10 12 13 ; do
    timestamp=$(date '+%Y-%m-%d-%H-%M-%S')
    echo "Training scan$i"
    
    if ! python train.py \
            -s "data/test-replica/scan$i/" \
            -o "output/test-replica/scan${i}_${timestamp}/" \
            --sfm_config posed \
            --use_view_config \
            --config_view_num 5 \
            --select_inpaint_num 10 \
            --use_refine_depth \
            --use_downsample_gaussians \
            --dense_supervision; then
        failed_scans="$failed_scans $i"
    fi
    echo "----------------------------------------"
done

if [ -n "$failed_scans" ]; then
    failed_scans_trimmed=$(echo "$failed_scans" | sed 's/^ //')
    echo "Following scans failed: $failed_scans_trimmed"
    exit 1
else
    echo "All scans succeeded!"
fi