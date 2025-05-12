#!/bin/bash

python 2d-gaussian-splatting/trace_object_with_sam2.py \
    -s output/replica-5-views/scan6/mast3r_sfm \
    -m output/replica-5-views/scan6/free_gaussians \
    --iteration 14000 \
    --output_root_path output/replica-5-views/scan6/free_gaussians/trace-sam2

