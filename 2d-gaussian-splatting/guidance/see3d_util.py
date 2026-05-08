import os
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from glob import glob
import re
import shutil
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor, AutoProcessor, AutoModel
from See3D_modules.mv_diffusion import mvdream_diffusion_model
from See3D_modules.mv_diffusion_SR import mvdream_diffusion_model as mvdream_diffusion_model_SR
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import gc
import time
import json
from skimage.metrics import structural_similarity as ssim
import lpips


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
#clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
pick_processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
pick_model_name = "yuvalkirstain/PickScore_v1"
pick_processor = AutoProcessor.from_pretrained(pick_processor_name)
pick_model = AutoModel.from_pretrained(pick_model_name).eval().to(device)
loss_fn = lpips.LPIPS(net='alex', spatial=False).cuda()


def compute_stability_score(frame_idx, output_dirs, lpips_model):

    imgs_pil = []
    imgs_tensors = []
    
    for d in output_dirs:
    
        img_path = os.path.join(d, f"SR_predict_warp_frame{frame_idx:06d}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(d, f"predict_warp_frame{frame_idx:06d}.png")
        
        if not os.path.exists(img_path):
            continue
        
        img = Image.open(img_path).convert('RGB')
        imgs_pil.append(np.array(img))
        
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        imgs_tensors.append(t.unsqueeze(0).cuda() * 2.0 - 1.0)

    if len(imgs_tensors) < 2:
        return None

    lpips_dists = []
    ssim_scores = []

    for i in range(len(imgs_tensors)):
        for j in range(i + 1, len(imgs_tensors)):
            # A. 计算 LPIPS
            with torch.no_grad():
                dist = lpips_model(imgs_tensors[i], imgs_tensors[j])
                lpips_dists.append(dist.item())
            
            s_val = ssim(imgs_pil[i], imgs_pil[j], channel_axis=2, data_range=255)
            ssim_scores.append(s_val)

    avg_lpips = float(np.mean(lpips_dists))
    avg_ssim = float(np.mean(ssim_scores))

    lpips_perceptual_score = np.exp(-avg_lpips * 6.5) 
    
    stability_score = 0.6 * lpips_perceptual_score + 0.4 * avg_ssim

    return {
        "avg_lpips": avg_lpips,
        "avg_ssim": avg_ssim,
        "stability_score": float(stability_score)
    }


def normalize_pickscore(score, center=-2.5, scale=0.5):
    import math
    try:
        return 1 / (1 + math.exp(-(score - center) / scale))
    except OverflowError:
        return 1.0 if score > center else 0.0


def compute_pick_scores(image_paths, model, processor, device):
    images = [Image.open(p).convert('RGB') for p in image_paths]
    

    pos_prompt = "A high-quality architectural photo with rigid geometry, straight edges, and good structural integrity."
    neg_prompt = "A distorted photo with warped architecture, melted furniture, structural collapse, and unrealistic twisted geometry."
    
    prompts = [pos_prompt, neg_prompt]

    # 2. 预处理图像和文本
    inputs = processor(
        images=images,
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        image_embs = model.get_image_features(pixel_values=inputs.pixel_values)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(input_ids=inputs.input_ids)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        logit_scale = model.logit_scale.exp()
        
        raw_scores = logit_scale * torch.matmul(text_embs, image_embs.T)
        
        pos_scores = raw_scores[0]
        neg_scores = raw_scores[1]
        
        final_scores = pos_scores - neg_scores * 1.2
        
    return final_scores.cpu().tolist()

class See3D(nn.Module):
    def __init__(
        self,
        device,
        base_model_path='./checkpoint/MVD_weights/',
        model_type='sparse',                                # single or sparse
        use_SR=False,
        seed=12345,
    ):
        super().__init__()

        self.device = device
        mv_unet_path = os.path.join(base_model_path, f"unet/{model_type}/ema-checkpoint")

        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        self.rgb_model = mvdream_diffusion_model(base_model_path, mv_unet_path, tokenizer, seed=seed)
        if use_SR:
            self.rgb_model_SR = mvdream_diffusion_model_SR(base_model_path, mv_unet_path, tokenizer, seed=seed)

    def PIL2tensor(self, height, width, num_frames, masks, warps, logicalNot=False):
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
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.GIF', '*.BMP', '*.TIFF', '*.WEBP']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(folder_path, ext)))

        image_names = [os.path.basename(file) for file in image_files]
        
        return image_names
    
    def load_ref_images(self, folder_path, height_mvd, width_mvd):
        temp_image_names = self.get_image_files(folder_path)
        temp_img_path = os.path.join(folder_path, temp_image_names[0])
        temp_img = Image.open(temp_img_path)
        width_ref, height_ref = temp_img.size

        ref_images = []
        ref_image_names = []
        if height_ref != height_mvd or width_ref != width_mvd:
            # split into two images
            if height_ref > width_ref:
                # resize width_ref to width_mvd
                height_tgt = int(height_ref * width_mvd / width_ref)
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name))
                    img = img.resize((width_mvd, height_tgt))

                    # split into two images
                    img_top = img.crop((0, 0, width_mvd, height_mvd))
                    img_bottom = img.crop((0, height_tgt-height_mvd, width_mvd, height_tgt))

                    img_name_top = image_name.split('.')[0] + '_top.png'
                    img_name_bottom = image_name.split('.')[0] + '_bottom.png'

                    ref_images.append(img_top)
                    ref_images.append(img_bottom)
                    ref_image_names.append(img_name_top)
                    ref_image_names.append(img_name_bottom)
            elif width_ref > height_ref:
                # resize height_ref to height_mvd
                width_tgt = int(width_ref * height_mvd / height_ref)
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name))
                    img = img.resize((width_tgt, height_mvd))

                    # split into two images
                    img_left = img.crop((0, 0, width_mvd, height_mvd))
                    img_right = img.crop((width_tgt-width_mvd, 0, width_tgt, height_mvd))

                    img_name_left = image_name.split('.')[0] + '_left.png'
                    img_name_right = image_name.split('.')[0] + '_right.png'

                    ref_images.append(img_left)
                    ref_images.append(img_right)
                    ref_image_names.append(img_name_left)
                    ref_image_names.append(img_name_right)
            else:
                # resize both height_ref and width_ref to height_mvd and width_mvd
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name))
                    img = img.resize((width_mvd, height_mvd))

                    ref_images.append(img)
                    ref_image_names.append(image_name)

        return ref_images, ref_image_names

    def inpainting(self, source_imgs_dir, warp_root_dir, output_root_dir, super_resolution=False):

        os.makedirs(output_root_dir, exist_ok=True)

        height_mvd = 512
        width_mvd = 512
        masks_infer = []
        warps_infer = []
        input_names = []

        # load source images
        gt_num_b = 0
        mask2 = np.ones((height_mvd, width_mvd), dtype=np.float32)

        # image_names_ref = self.get_image_files(source_imgs_dir)
        # fimage = Image.open(os.path.join(source_imgs_dir, image_names_ref[0]))
        # (width, height)= fimage.size

        ref_images, ref_image_names = self.load_ref_images(source_imgs_dir, height_mvd, width_mvd)

        for imn, ref_img in zip(ref_image_names, ref_images):
            masks_infer.append(Image.fromarray(np.repeat(np.expand_dims(np.round(mask2*255.).astype(np.uint8),axis=2),3,axis=2)).resize((width_mvd, height_mvd)))
            warps_infer.append(ref_img)
            input_names.append(imn)
            gt_num_b = gt_num_b + 1

        # load warp images and masks
        image_files = glob(os.path.join(warp_root_dir, "warp_*"))
        image_names = [os.path.basename(image) for image in image_files]
        image_names.sort()

        fimage = Image.open(os.path.join(warp_root_dir, image_names[0]))
        (width, height)= fimage.size

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
            images_predict[jj].resize((width, height)).save(os.path.join(output_root_dir,"predict_{}".format(images_predict_names[jj])))

        print(f'end inpainting, result saved in {output_root_dir}')

        if super_resolution:
            print('start SR inpainting')
            del self.rgb_model
            gc.collect()
            torch.cuda.empty_cache()

            masks_infer_SR = []
            warps_infer_SR = []
            mask2 = np.ones((height_mvd*2,width_mvd*2), dtype=np.float32)

            ref_images, ref_image_names = self.load_ref_images(source_imgs_dir, height_mvd, width_mvd)

            for imn, ref_img in zip(ref_image_names, ref_images):
                masks_infer_SR.append(Image.fromarray(np.repeat(np.expand_dims(np.round(mask2*255.).astype(np.uint8),axis=2),3,axis=2)).resize((width_mvd, height_mvd)))
                warps_infer_SR.append(ref_img)

            for i in range(len(images_predict)):
                masks_infer_SR.append(masks_infer[i])
                warps_infer_SR.append(images_predict[i])

            images_predict = []
            images_predict_names = []
            # grounp_size = min((len(masks_infer_SR) + 5)//2,50)
            grounp_size = (len(masks_infer_SR) + 3) // 2
            # grounp_size = (len(masks_infer_SR) + 3)
            print('grounp_size:',grounp_size)
            for i in range(0, len(masks_infer_SR[gt_num_b:]), grounp_size):
                if(len(images_predict)!=0):
                    masks_infer_batch = masks_infer_SR[:gt_num_b] + [masks_infer_batch[len(masks_infer_batch)//2]] + [masks_infer_batch[-1]] + masks_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                    warp_infer_batch = warps_infer_SR[:gt_num_b] + [images_predict[len(images_predict)//2]] + [images_predict[-1]] + warps_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                    input_names_batch = input_names[:gt_num_b] + [input_names_batch[len(masks_infer_batch)//2]] + [input_names_batch[-1]] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                else:
                    masks_infer_batch = masks_infer_SR[:gt_num_b] + masks_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                    warp_infer_batch = warps_infer_SR[:gt_num_b] + warps_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                    input_names_batch = input_names[:gt_num_b] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]

                
                prompt, batch = self.PIL2tensor(height_mvd*2,width_mvd*2,len(masks_infer_batch),masks_infer_batch,warp_infer_batch)
                if(len(images_predict)!=0):
                    images_predict_batch = self.rgb_model_SR.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd*2,width_mvd*2,gt_num_frames=gt_num_b,output_type='pil')
                    for jj in range(gt_num_b+2,len(images_predict_batch)):
                        images_predict.append(images_predict_batch[jj])
                        images_predict_names.append(input_names_batch[jj])
                else:
                    images_predict_batch = self.rgb_model_SR.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd*2,width_mvd*2,gt_num_frames=gt_num_b,output_type='pil')
                    for jj in range(gt_num_b,len(images_predict_batch)):
                        images_predict.append(images_predict_batch[jj])
                        images_predict_names.append(input_names_batch[jj])
                gc.collect()
                torch.cuda.empty_cache()


            for jj in range(len(images_predict)):
                images_predict[jj].resize((width_mvd*2, height_mvd*2)).save(os.path.join(output_root_dir,"SR_predict_{}".format(images_predict_names[jj])))

            print(f'end SR inpainting, result saved in {output_root_dir}')
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ref_imgs_dir', type=str, required=True)
    parser.add_argument('--warp_root_dir', type=str, required=True) # stageX/select-gs
    parser.add_argument('--output_root_dir', type=str, required=True)
    parser.add_argument('--use_SR', action='store_true')
    parser.add_argument('--see3d_stage', type=int, default=1) 
    args = parser.parse_args()

    # --- 1. Init Models and Features ---
    print("Initializing Models...")
    source_imgs_dir = args.ref_imgs_dir
    warp_root_dir = args.warp_root_dir
    output_root_dir = args.output_root_dir
    output_dirs = [output_root_dir, output_root_dir + "_s1", output_root_dir + "_s2"]
    seeds = [1234, 2345, 3456]


    # --- 2. Multi-Seed Inference ---
    t1 = time.time()
    for seed, out_dir in zip(seeds, output_dirs):
        print(f"\n>>> Running Inference | Seed: {seed}")
        see3d = See3D(device='cuda', use_SR=args.use_SR, seed=seed)
        see3d.inpainting(source_imgs_dir, warp_root_dir, out_dir, args.use_SR)
        del see3d; gc.collect(); torch.cuda.empty_cache()

    # --- 3. Active Vision Greedy Selection ---
    print("\n>>> Evaluating views with Active Vision Greedy Selection...")
    stage_dir = os.path.dirname(output_root_dir)
    matrix_path = os.path.join(stage_dir, "candidate_vis_matrix.npy")
    diag_dir = os.path.join(stage_dir, "diagnosis_mags")
    os.makedirs(diag_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 50)
    except:
        print("font import false, use default font")
        font = ImageFont.load_default()
    
    candidate_pool = []
    warp_files = glob(os.path.join(output_root_dir, "predict_warp_frame*.png"))
    # Ensure indices are sorted to match visibility matrix rows
    frame_indices = sorted({int(re.search(r'frame(\d+)', os.path.basename(f)).group(1)) for f in warp_files})
    
    for idx in frame_indices:
        idx_str = f"{idx:06d}"
        stab_res = compute_stability_score(idx, output_dirs, loss_fn)
        if not stab_res: continue
        
        img_prefix = "SR_predict_" if args.use_SR else "predict_"
        seed_image_paths = [os.path.join(d, f"{img_prefix}warp_frame{idx_str}.png") for d in output_dirs]
        seed_scores = compute_pick_scores(seed_image_paths, pick_model, pick_processor, device)
        
        best_seed_idx = int(np.argmax(seed_scores))
        max_pick_score = seed_scores[best_seed_idx]
        #best_seed_idx = int(np.argmax(seed_clips))
        norm_pick_score = normalize_pickscore(max_pick_score)

        ###### ab study
        quality_score = 1.0 * stab_res["stability_score"] + 1.0 * norm_pick_score
        candidate_pool.append({"idx": idx, "quality_score": quality_score, "seed_idx": best_seed_idx})
        ori_warp_path = os.path.join(args.warp_root_dir, f"warp_frame{idx_str}.png")
        if os.path.exists(ori_warp_path):
            img_ori = Image.open(ori_warp_path).convert('RGB')
            w, h = img_ori.size
            diag_img = Image.new('RGB', (w * 4, h + 160), (255, 255, 255))
            diag_img.paste(img_ori, (0, 0))
            
            for i, d in enumerate(output_dirs):
                s_path = os.path.join(d, f"{img_prefix}warp_frame{idx_str}.png")
                if os.path.exists(s_path):
                    s_img = Image.open(s_path).convert('RGB')
                    diag_img.paste(s_img, (w * (i + 1), 0))

            draw = ImageDraw.Draw(diag_img)
            txt = f"Frame: {idx_str} | Stab: {stab_res['stability_score']:.4f} | Final Quality: {quality_score:.4f}\n"
            txt += f"CLIP: S0={normalize_pickscore(seed_scores[0]):.2f}, S1={normalize_pickscore(seed_scores[1]):.2f}, S2={normalize_pickscore(seed_scores[2]):.2f} | Best: Seed_{best_seed_idx}"
            draw.text((20, h + 15), txt, fill=(0, 0, 0), font=font)
            
            diag_img.save(os.path.join(diag_dir, f"diag_frame{idx_str}.jpg"), quality=80)

    vis_matrix = np.load(matrix_path) if os.path.exists(matrix_path) else None
    final_selection = []

    if vis_matrix is not None:
        num_candidates, num_pts = vis_matrix.shape
        coverage_counts = np.zeros(num_pts, dtype=np.int32)
        selected_p_indices = []
        if args.see3d_stage <= 2:
            reference_count = num_pts * 0.1
        else :
            reference_count = num_pts * 0.06

        for _ in range(min(11, len(candidate_pool))):
            best_score = -1e9
            best_p_idx = -1

            for p_idx, cand in enumerate(candidate_pool):
                if p_idx in selected_p_indices: continue
                
                view_vis = vis_matrix[p_idx]
                discovery_gain = np.sum((coverage_counts == 0) & (view_vis == 1))
                reconstruction_gain = np.sum((coverage_counts == 1) & (view_vis == 1))
                
                if args.see3d_stage <= 3:
                    max_iou_with_selected = 0
                    if len(selected_p_indices) > 0:
                        for prev_idx in selected_p_indices:
                            prev_vis = vis_matrix[prev_idx]
                            inter = (view_vis & prev_vis).sum()
                            union = (view_vis | prev_vis).sum()
                            iou = inter / union if union > 0 else 0
                            max_iou_with_selected = max(max_iou_with_selected, iou)

                    spatial_penalty = 1.0
                    if max_iou_with_selected > 0.55:
                        spatial_penalty = 0.01 
                    
                    geom_score = (1.5 * discovery_gain) + (1.0 * reconstruction_gain)
                    norm_geom = geom_score / reference_count
                    combined = (0.5 * cand['quality_score'] + 0.5 * norm_geom) * spatial_penalty

                if combined > best_score:
                    best_score = combined
                    best_p_idx = p_idx
            
            if best_p_idx != -1:
                selected_p_indices.append(best_p_idx)
                coverage_counts += vis_matrix[best_p_idx]
                final_selection.append({
                    "idx": candidate_pool[best_p_idx]['idx'], 
                    "score": best_score, 
                    "seed_idx": candidate_pool[best_p_idx]['seed_idx']
                })
        final_selection.sort(key=lambda x: x['idx'])
    else:
        # Fallback to quality-only
        candidate_pool.sort(key=lambda x: x['quality_score'], reverse=True)
        final_selection = sorted(candidate_pool[:10], key=lambda x: x['idx'])

# --- 3.2 Mark Selected Images in Diagnosis Folder ---
    print(">>> Marking selected views in diagnosis folder...")
    for item in final_selection:
        old_diag_path = os.path.join(diag_dir, f"diag_frame{item['idx']:06d}.jpg")
        new_diag_path = os.path.join(diag_dir, f"diag_frame{item['idx']:06d}_(selected).jpg")
        if os.path.exists(old_diag_path):
            os.rename(old_diag_path, new_diag_path)

    # --- 4. Synchronization and File Ops ---
    print(f"\n>>> Synchronizing {len(final_selection)} selected views...")
    inpaint_target_dir = os.path.join(stage_dir, "select-gs-inpainted")
    cur_npz_path = os.path.join(stage_dir, f"stage{args.see3d_stage}_see3d_cameras.npz")
    
    original_npz = dict(np.load(cur_npz_path, allow_pickle=True))
    new_npz = {'train_views': original_npz.get('train_views')}

    tmp_inpaint = os.path.join(stage_dir, "tmp_inpaint")
    tmp_select_gs = os.path.join(stage_dir, "tmp_select_gs")
    os.makedirs(tmp_inpaint, exist_ok=True)
    os.makedirs(tmp_select_gs, exist_ok=True)

    file_templates = ["alpha_{suffix}.npy", "alpha_mask_frame{suffix}.png", "alpha_warp_frame{suffix}.png", 
                      "depth_frame{suffix}.tiff", "mask_frame{suffix}.png", "ori_warp_frame{suffix}.png", "warp_frame{suffix}.png"]

    for new_idx, item in enumerate(final_selection):
        old_s, new_s = f"{item['idx']:06d}", f"{new_idx:06d}"
        in_p = "SR_predict_" if args.use_SR else "predict_"
        
        # Move Best Inpaint
        shutil.copy(os.path.join(output_dirs[item['seed_idx']], f"{in_p}warp_frame{old_s}.png"), 
                    os.path.join(tmp_inpaint, f"predict_warp_frame{new_s}.png"))
        
        # Move Geometry Aux Files
        for temp in file_templates:
            src_aux = os.path.join(args.warp_root_dir, temp.format(suffix=old_s))
            if os.path.exists(src_aux):
                shutil.copy(src_aux, os.path.join(tmp_select_gs, temp.format(suffix=new_s)))

        # Sync Camera NPZ
        for key in ['R', 'T', 'FoVx', 'FoVy', 'image_width', 'image_height']:
            if f"{key}_{old_s}" in original_npz:
                new_npz[f"{key}_{new_s}"] = original_npz[f"{key}_{old_s}"]

    # --- 5. Atomic Refresh of Directories ---
    def safe_refresh(target, source):
        os.makedirs(target, exist_ok=True)
        # Clear target content first instead of removing the directory
        for item in os.listdir(target):
            item_path = os.path.join(target, item)
            if os.path.isfile(item_path): os.remove(item_path)
            elif os.path.isdir(item_path): shutil.rmtree(item_path)
        # Move new files
        for item in os.listdir(source):
            shutil.move(os.path.join(source, item), os.path.join(target, item))
        shutil.rmtree(source)

    safe_refresh(inpaint_target_dir, tmp_inpaint)
    safe_refresh(args.warp_root_dir, tmp_select_gs)

    new_npz['n_views'] = len(final_selection)
    np.savez(cur_npz_path, **new_npz)

    print(f"✅ Stage {args.see3d_stage} refinement complete. Results in: {inpaint_target_dir}")