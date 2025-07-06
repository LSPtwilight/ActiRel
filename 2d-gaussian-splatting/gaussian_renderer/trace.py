import torch
import math
from diff_id_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def trace(viewpoint_camera, pc : GaussianModel,
          id_masks,                 # need to be continue int, like 0, 1, 2 ...
          pipe, 
          bg_color : torch.Tensor,  scaling_modifier = 1.0, override_color = None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    id_masks=id_masks.to(torch.int).cuda()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=False           # NOTE: not need use contrastive feature, directly use consistent id mask
        # pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    num_class = id_masks.max()
    weights = torch.zeros((pc.get_opacity.shape[0], num_class+1), dtype=torch.int, device='cuda')
    
    rasterizer.trace(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        weights=weights,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        id_masks=id_masks,
        num_class=num_class
    )
    return weights

def get_weights_by_all_views(gaussians, viewpoints, pipe, background, unseen_value=-1, threshold=5):
    with torch.no_grad():
        weights = torch.zeros((gaussians.get_opacity.shape[0], len(viewpoints)), dtype=torch.int).cuda()
        for idx, view in enumerate(viewpoints):
            id_masks = view.instance_map.to(torch.int)        # NOTE: use gt mask or other multi-view consistent id mask
            w = trace(view, gaussians, id_masks, pipe, background)                              # shape: [N_Gaussian, N_ID], represent each gaussian trace result in this view
            torch.cuda.synchronize()

            unseen_mask = (w.sum(-1) < threshold)
            w = torch.argmax(w, dim=-1)
            w[unseen_mask] = unseen_value                               # this gaussian is not seen in this view
            weights[:,idx] = w
    return weights                                                      # shape: [N_Gaussian, N_View]

def get_weights_by_single_view(gaussians, viewpoint, pipe, background, unseen_value=-1, threshold=5):
    with torch.no_grad():
        weights = torch.zeros((gaussians.get_opacity.shape[0], 1), dtype=torch.int).cuda()
        id_masks = viewpoint.instance_map.to(torch.int)        # NOTE: use gt mask or other multi-view consistent id mask
        w = trace(viewpoint, gaussians, id_masks, pipe, background)                              # shape: [N_Gaussian, N_ID], represent each gaussian trace result in this view
        torch.cuda.synchronize()
        
        unseen_mask = (w.sum(-1) < threshold)
        w = torch.argmax(w, dim=-1)
        w[unseen_mask] = unseen_value                                   # this gaussian is not seen in this view
        weights[:,0] = w
    return weights                                                      # shape: [N_Gaussian, 1]

def get_weights_by_single_view_with_mask(gaussians, viewpoint, pipe, background, obj_mask, unseen_value=-1, threshold=5):
    with torch.no_grad():
        weights = torch.zeros((gaussians.get_opacity.shape[0], 1), dtype=torch.int).cuda()
        w = trace(viewpoint, gaussians, obj_mask, pipe, background)                              # shape: [N_Gaussian, N_ID], represent each gaussian trace result in this view
        torch.cuda.synchronize()
        
        unseen_mask = (w.sum(-1) < threshold)
        w = torch.argmax(w, dim=-1)
        w[unseen_mask] = unseen_value                                   # this gaussian is not seen in this view
        weights[:,0] = w
    return weights                                                      # shape: [N_Gaussian, 1]
