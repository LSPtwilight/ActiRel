#!/bin/bash

data_path=/home/nijunfeng/mycode/project/gs-recon/priorgs/data/replica/scan6
source_path=/home/nijunfeng/mycode/project/gs-recon/priorgs/output/replica-5-views/scan6/mast3r_sfm
root_path=/home/nijunfeng/mycode/project/gs-recon/priorgs/output/replica-5-views/scan6/free_gaussians
model_path=$root_path
iteration=7000

source_imgs_dir=/home/nijunfeng/mycode/project/gs-recon/priorgs/mycode/training_5views/scan6

render_method=gs
warp_root_dir=$root_path/see3d_render/select-$render_method
output_root_dir=$root_path/see3d_render/select-$render_method-inpainted

python 2d-gaussian-splatting/render_novel_views.py \
    --model_path $root_path \
    --iteration $iteration \
    --data_path $data_path \
    --output_root_path $warp_root_dir

python 2d-gaussian-splatting/guidance/see3d_util.py \
    --source_imgs_dir $source_imgs_dir \
    --warp_root_dir $warp_root_dir \
    --output_root_dir $output_root_dir

python 2d-gaussian-splatting/train_hier_gaussian.py \
    -s $source_path \
    -m $model_path \
    --warp_root_path $warp_root_dir \
    --inpaint_root_path $output_root_dir \
    --output_root_path $root_path/see3d_render/$render_method-trained_hier_model
