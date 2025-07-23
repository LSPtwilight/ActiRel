#!/bin/bash

export http_proxy=""
export https_proxy=""
export all_proxy=""
export no_proxy="localhost,127.0.0.1"


# get views
for i in 6
do
    TIMESTAMP=$(date +"%Y-%m-%d")
    echo "Training scan$i"
    python train.py -s data/test-replica/scan$i/ -o output/test-replica/scan${i}_${TIMESTAMP}/ --sfm_config posed --use_view_config --config_view_num 5 --select_inpaint_num 10 --use_refine_depth
    # python train.py -s data/replica/scan$i/ -o output/replica/scan${i}_${TIMESTAMP}/ --sfm_config posed --use_view_config --config_view_num 6 --select_inpaint_num 10 --use_refine_depth
    echo "Finished training scan$i"
done
echo "Finished training all scans"


