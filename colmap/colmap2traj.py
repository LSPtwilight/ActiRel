import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from matcha.dm_utils.dataset_readers import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat


data_path = '/home/nijunfeng/mycode/project/gs-recon/MAtCha/output/replica/scan3/mast3r_sfm/all-sparse/scaled'
save_txt_path = os.path.join(data_path, 'traj.txt')

src_camera_data = read_intrinsics_binary(f'{data_path}/cameras.bin')
src_image_data = read_extrinsics_binary(f'{data_path}/images.bin')

src_intrinsics = {}
c2w_list = []
for k in src_image_data:
    img_name = src_image_data[k].name

    cam = src_camera_data[src_image_data[k].camera_id]
    if cam.model == 'PINHOLE':
        fx,fy,cx,cy = cam.params
    else:
        raise NotImplementedError('Only PINHOLE model is supported for now.')
    src_intrinsics[img_name] = np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ])

    T = np.eye(4)
    T[:3,:3] = qvec2rotmat(src_image_data[k].qvec)
    T[:3,3] = src_image_data[k].tvec
    
    c2w = np.linalg.inv(T)
    c2w_list.append(c2w)

with open(save_txt_path, 'w') as f:
    for c2w in c2w_list:
        line = ' '.join([str(val) for val in c2w.flatten()])
        f.write(line + '\n')

print(f"Saved to {save_txt_path}")



