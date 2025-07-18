import os
import shutil
import numpy as np
from PIL import Image
from Difix3D_modules.pipeline_difix import DifixPipeline
from diffusers.utils import load_image

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # load difix3d pipeline (without reference image)
    pipe = DifixPipeline.from_pretrained("./checkpoint/difix")
    pipe.to("cuda")

    file_list = os.listdir(input_dir)
    for file_name in file_list:
        if 'warp_frame' not in file_name:
            src_path = os.path.join(input_dir, file_name)
            dst_path = os.path.join(output_dir, file_name)
            shutil.copy(src_path, dst_path)
    print("Copy Done")

    # process difix3d
    img_list = [img_name for img_name in file_list if 'ori_warp_frame' in img_name]
    img_list.sort()
    for img_name in img_list:

        input_path = os.path.join(input_dir, img_name)
        input_image = load_image(input_path)
        prompt = "remove degradation"

        output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
        output_image.save(os.path.join(output_dir, img_name))

        # save warp frame
        ori_warp_map = np.array(output_image)
        ori_warp_map = ori_warp_map.astype(np.uint8)
        
        mask_name = img_name.replace('ori_warp_frame', 'mask_frame')
        mask_path = os.path.join(input_dir, mask_name)
        mask_img = Image.open(mask_path)
        mask_img = np.array(mask_img) / 255

        warp_map = ori_warp_map * mask_img[:,:,None]
        warp_map = Image.fromarray(warp_map.astype(np.uint8))
        save_warp_name = img_name.replace('ori_warp_frame', 'warp_frame')
        warp_map.save(os.path.join(output_dir, save_warp_name))

        print(f"Processed {img_name}")

    print("Difix3D Done")
