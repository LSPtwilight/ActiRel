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
    python train-AB-E1.py -s data/replica/scan$i/ -o output/replica-AB-E1/scan${i}_${TIMESTAMP}/ --sfm_config posed --use_view_config --config_view_num 5 --select_inpaint_num 20
    echo "Finished training scan$i"
done
echo "Finished training all scans"


