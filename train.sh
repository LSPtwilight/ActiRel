#!/bin/bash

# for i in 1
# do
#     echo "Training scan$i"
#     python train.py -s data/replica-unposed/scan$i/ -o output/replica-unposed/scan$i/ --sfm_config unposed
#     echo "Finished training scan$i"
# done
# echo "Finished training all scans"

# use posed replica dataset
# for i in 1 2 3 4 5 6 7 8
for i in 6
do
    echo "Training scan$i"
    python train.py -s data/replica/scan$i/ -o output/replica/scan$i/ --sfm_config posed --use_view_config
    echo "Finished training scan$i"
done
echo "Finished training all scans"


# for i in 6
# do
#     echo "Training scan$i"
#     python train.py -s data/replica-test/scan$i/ -o output/replica-test/scan$i/ --sfm_config unposed
#     echo "Finished training scan$i"
# done
# echo "Finished training all scans"
