import os
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

    img_list = os.listdir(input_dir)
    img_list.sort()
    for img_name in img_list:

        input_path = os.path.join(input_dir, img_name)
        input_image = load_image(input_path)
        prompt = "remove degradation"

        output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
        output_image.save(os.path.join(output_dir, img_name))

        print(f"Processed {img_name}")

    print("Difix3D Done")
