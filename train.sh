#!/bin/bash

export http_proxy=""
export https_proxy=""
export all_proxy=""
export no_proxy="localhost,127.0.0.1"


# get views
for i in 13
do
    TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
    echo "Training scan$i"
    python train.py \
        -s data/test-replica/scan$i/ \
        -o output/test-replica/scan${i}_${TIMESTAMP}/ \
        --sfm_config posed \
        --use_view_config \
        --config_view_num 10 \
        --select_inpaint_num 10 \
        --use_refine_depth \
        --use_downsample_gaussians 
        #--dense_supervision 
    echo "Finished training scan$i"
done
echo "Finished training all scans"


