import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
from transformers import CLIPTextModel, CLIPTokenizer

from See3D_modules.mv_diffusion import mvdream_diffusion_model
from argparse import ArgumentParser


class See3D(nn.Module):
    def __init__(
        self,
        device,
        base_model_path='./checkpoint/MVD_weights/',
        model_type='sparse',                                # single or sparse
        seed=12345,
    ):
        super().__init__()

        self.device = device
        mv_unet_path = os.path.join(base_model_path, f"unet/{model_type}/ema-checkpoint")

        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        self.rgb_model = mvdream_diffusion_model(base_model_path, mv_unet_path, tokenizer, seed=seed)

    def PIL2tensor(self, height, width, num_frames, masks,warps, logicalNot=False):
        channels = 3
        pixel_values = torch.empty((num_frames, channels, height, width))
        condition_pixel_values = torch.empty((num_frames, channels, height, width))
        masks_pixel_values = torch.ones((num_frames, 1, height, width))
        
        # input_ids
        prompt = ''

        for i, img in enumerate(masks):
            img = masks[i]
            img = img.convert('L') # make sure channel 1
            img_resized = img.resize((width, height)) # hard code here
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

            # Normalize the image by scaling pixel values to [0, 1]
            img_normalized = img_tensor / 255
            mask_condition = (img_normalized > 0.9).float()
            
            masks_pixel_values[i] = mask_condition
        
        for i, img in enumerate(warps):
            # Resize the image and convert it to a tensor
            img_resized = img.resize((width, height)) # hard code here
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

            # Normalize the image by scaling pixel values to [-1, 1]
            img_normalized = img_tensor / 127.5 - 1

            img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images

            if(logicalNot):
                img_normalized = torch.logical_not(masks_pixel_values[i])*(-1) + masks_pixel_values[i]*img_normalized
            condition_pixel_values[i] = img_normalized
            
        return [prompt], {
                'conditioning_pixel_values': condition_pixel_values, # [-1,1]
                'masks': masks_pixel_values# [0,1]
                }
        
    def get_image_files(self, folder_path):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(folder_path, ext)))

        image_names = [os.path.basename(file) for file in image_files]
        
        return image_names

    def inpainting(self, source_imgs_dir, warp_root_dir, output_root_dir):

        os.makedirs(output_root_dir, exist_ok=True)

        height_mvd = 512
        width_mvd = 512
        masks_infer = []
        warps_infer = []
        input_names = []

        # load source images
        gt_num_b = 0
        mask2 = np.ones((height_mvd,width_mvd), dtype=np.float32)

        image_names_ref = self.get_image_files(source_imgs_dir)
        fimage = Image.open(os.path.join(source_imgs_dir, image_names_ref[0]))
        (width, height)= fimage.size
        for imn in image_names_ref:
            masks_infer.append(Image.fromarray(np.repeat(np.expand_dims(np.round(mask2*255.).astype(np.uint8),axis=2),3,axis=2)).resize((width_mvd, height_mvd)))
            warps_infer.append(Image.open(os.path.join(source_imgs_dir, imn)))
            input_names.append(imn)
            gt_num_b = gt_num_b + 1

        # load warp images and masks
        image_files = glob(os.path.join(warp_root_dir, "warp_*"))
        image_names = [os.path.basename(image) for image in image_files]
        image_names.sort()

        for ins in image_names:
            warps_infer.append(Image.open(os.path.join(warp_root_dir, ins))) 
            masks_infer.append(Image.open(os.path.join(warp_root_dir, ins.replace('warp','mask'))))
            input_names.append(ins)
        print('all inpainting sequence length:', len(warps_infer))

        # inpainting
        images_predict = []
        images_mask_p = []
        images_predict_names = []

        grounp_size = len(masks_infer)
        for i in range(0, len(masks_infer[gt_num_b:]), grounp_size):
            if(len(images_predict)!=0):
                masks_infer_batch = masks_infer[:gt_num_b] + [masks_infer_batch[-1]] + masks_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                warp_infer_batch = warps_infer[:gt_num_b] + [images_predict[-1]] + warps_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                input_names_batch = input_names[:gt_num_b] + [input_names_batch[len(masks_infer_batch)//2]] + [input_names_batch[-1]] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]
            else:
                masks_infer_batch = masks_infer[:gt_num_b] + masks_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                warp_infer_batch = warps_infer[:gt_num_b] + warps_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                input_names_batch = input_names[:gt_num_b] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]

            prompt, batch = self.PIL2tensor(height_mvd,width_mvd,len(masks_infer_batch),masks_infer_batch,warp_infer_batch,logicalNot=False)
            if(len(images_predict)!=0):
                images_predict_batch = self.rgb_model.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd,width_mvd,gt_num_frames=gt_num_b,output_type='pil')
                for jj in range(gt_num_b+1,len(images_predict_batch)):
                    images_predict.append(images_predict_batch[jj])
                    images_mask_p.append(batch['masks'][0][jj][0].cpu().numpy())
                    images_predict_names.append(input_names_batch[jj])
            else:
                images_predict_batch = self.rgb_model.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd,width_mvd,gt_num_frames=gt_num_b,output_type='pil')
                for jj in range(gt_num_b,len(images_predict_batch)):
                    images_predict.append(images_predict_batch[jj])
                    images_mask_p.append(batch['masks'][0][jj][0].cpu().numpy())
                    images_predict_names.append(input_names_batch[jj])
                
        for jj in range(len(images_predict)):
            images_predict[jj].resize((width, height)).save(os.path.join(output_root_dir,"predict_{}.jpg".format(images_predict_names[jj])))

        print(f'end inpainting, result saved in {output_root_dir}')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--source_imgs_dir', type=str)
    parser.add_argument('--warp_root_dir', type=str)
    parser.add_argument('--output_root_dir', type=str)
    args = parser.parse_args()

    source_imgs_dir = args.source_imgs_dir
    warp_root_dir = args.warp_root_dir
    output_root_dir = args.output_root_dir

    see3d = See3D(device='cuda')
    see3d.inpainting(source_imgs_dir=source_imgs_dir, warp_root_dir=warp_root_dir, output_root_dir=output_root_dir)


