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
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--need_cat_result", action='store_true', help='need to cat result')
    args = parser.parse_args()

    input_dir = args.input_dir
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        root_dir = os.path.dirname(input_dir)
        input_name = os.path.basename(input_dir)
        output_dir = os.path.join(root_dir, input_name + '_difix3d')
    os.makedirs(output_dir, exist_ok=True)

    if args.need_cat_result:
        root_dir = os.path.dirname(input_dir)
        input_name = os.path.basename(input_dir)
        cat_result_dir = os.path.join(root_dir, input_name + '_cat_result')
        os.makedirs(cat_result_dir, exist_ok=True)

    # load difix3d pipeline (without reference image)
    pipe = DifixPipeline.from_pretrained("./checkpoint/difix")
    pipe.to("cuda")

    img_list = os.listdir(input_dir)
    img_list = [img_name for img_name in img_list if img_name.endswith('.png')]
    img_list.sort()
    for img_name in img_list:

        input_path = os.path.join(input_dir, img_name)
        input_image = load_image(input_path)
        prompt = "remove degradation"

        output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
        output_image.save(os.path.join(output_dir, img_name))
        if args.need_cat_result:
            ori_img = Image.open(os.path.join(input_dir, img_name))
            refine_img = Image.open(os.path.join(output_dir, img_name))
            cat_img = Image.new('RGB', (ori_img.width * 2 + 5, ori_img.height))
            cat_img.paste(ori_img, (0, 0))
            cat_img.paste(refine_img, (ori_img.width + 5, 0))
            cat_img.save(os.path.join(cat_result_dir, img_name))

        print(f"Processed {img_name}")

    print("Difix3D Done")
