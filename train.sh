#!/bin/bash

# eval all posed replica dataset
for i in 1 2 3 4 5 6 7 8
do
    for test_str in 't1' 't2'
    do
        config_view_num=5
        select_inpaint_num=20
        echo "Training scan$i-$test_str with $config_view_num views"
        python train.py -s data/replica/scan$i/ -o output/replica-$config_view_num-views/scan$i-$test_str/ --sfm_config posed --use_view_config --config_view_num $config_view_num --select_inpaint_num $select_inpaint_num
        echo "Finished training scan$i-$test_str with $config_view_num views"
    done
done
echo "Finished training all scans"


# for i in 6
# do
#     config_view_num=5
#     select_inpaint_num=20
#     echo "Training scan$i with $config_view_num views"
#     python train.py -s data/replica/scan$i/ -o output/replica-$config_view_num-views/scan$i/ --sfm_config posed --use_view_config --config_view_num $config_view_num --select_inpaint_num $select_inpaint_num
#     echo "Finished training scan$i with $config_view_num views"
# done
# echo "Finished training all scans"

# for i in 6
# do
#     echo "Training scan$i"
#     python train.py -s data/replica-see3d/scan$i/ -o output/replica-see3d/scan$i/ --sfm_config posed
#     echo "Finished training scan$i"
# done
# echo "Finished training all scans"
