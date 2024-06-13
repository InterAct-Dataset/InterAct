# 
# ## Extract Poses from Amass Dataset

# 
# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook
# %matplotlib inline

import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
from markerset import markerset_smplh, markerset_smplx

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 
# ### Please remember to download the following subdataset from AMASS website: https://amass.is.tue.mpg.de/download.php. Note only download the <u>SMPL+H G</u> data.
# * ACCD (ACCD)
# * HDM05 (MPI_HDM05)
# * TCDHands (TCD_handMocap)
# * SFU (SFU)
# * BMLmovi (BMLmovi)
# * CMU (CMU)
# * Mosh (MPI_mosh)
# * EKUT (EKUT)
# * KIT  (KIT)
# * Eyes_Janpan_Dataset (Eyes_Janpan_Dataset)
# * BMLhandball (BMLhandball)
# * Transitions (Transitions_mocap)
# * PosePrior (MPI_Limits)
# * HumanEva (HumanEva)
# * SSM (SSM_synced)
# * DFaust (DFaust_67)
# * TotalCapture (TotalCapture)
# * BMLrub (BioMotionLab_NTroje)
# 
# ### Unzip all datasets. In the bracket we give the name of the unzipped file folder. Please correct yours to the given names if they are not the same.

# 
# ### Place all files under the directory **./amass_data/**. The directory structure shoud look like the following:  
# ./amass_data/  
# ./amass_data/ACCAD/  
# ./amass_data/BioMotionLab_NTroje/  
# ./amass_data/BMLhandball/  
# ./amass_data/BMLmovi/   
# ./amass_data/CMU/  
# ./amass_data/DFaust_67/  
# ./amass_data/EKUT/  
# ./amass_data/Eyes_Japan_Dataset/  
# ./amass_data/HumanEva/  
# ./amass_data/KIT/  
# ./amass_data/MPI_HDM05/  
# ./amass_data/MPI_Limits/  
# ./amass_data/MPI_mosh/  
# ./amass_data/SFU/  
# ./amass_data/SSM_synced/  
# ./amass_data/TCD_handMocap/  
# ./amass_data/TotalCapture/  
# ./amass_data/Transitions_mocap/  
# 
# **Please make sure the file path are correct, otherwise it can not succeed.**

# 
# Choose the device to run the body model on.
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 

import smplx

MODEL_PATH = './body_models'


smpl_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                          gender="male",
                          use_pca=False,
                          ext='pkl').cuda()

smpl_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                          gender="female",
                          use_pca=False,
                          ext='pkl').cuda()

smpl_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
                          gender="neutral",
                          use_pca=False,
                          ext='pkl').cuda()

smpl = {'male': smpl_model_male, 'female': smpl_model_female, 'neutral': smpl_model_neutral}




paths = []
folders = []
dataset_names = []
for root, dirs, files in os.walk('./dataset/behave/sequences_seg'):
    folders.append(root)
    for name in files:
        dataset_name = root.split('/')[2]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        paths.append(os.path.join(root, files[0]))




#
save_root = './dataset/behave/markers'
save_folders = [folder.replace('./dataset/behave/sequences_seg', save_root) for folder in folders]
for folder in save_folders:
    os.makedirs(folder, exist_ok=True)
group_path = [[path for path in paths if name in path] for name in dataset_names]

# %%

def behave_to_marker(src_path, save_path):
    

    src_path_obj = src_path.replace('human.npz', 'object.npz')

    bdata = np.load(src_path, allow_pickle=True)
    bdata_obj = np.load(src_path_obj, allow_pickle=True)
    fps = 30
    frame_times = bdata['trans'].shape[0]
  
#     print(frame_number)
#     print(fps)
    poses, betas, trans, gender = bdata['poses'], bdata['betas'], bdata['trans'], bdata['gender']
    smpl_model = smpl[str(gender)]
    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).to(comp_device).float(),
                                global_orient=torch.from_numpy(poses[:, :3]).to(comp_device).float(),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).to(comp_device).float(),
                                right_hand_pose=torch.from_numpy(poses[:, 111:]).to(comp_device).float(),
                                betas=torch.from_numpy(betas).to(comp_device).float(),
                                transl=torch.from_numpy(trans).to(comp_device).float())
    
                             
    verts = smplx_output.vertices
    markers = verts[:,markerset_smplh].detach().cpu().numpy()
    
    np.save(save_path, markers)

    # process obj pose data
    angle, trans = bdata_obj['angles'], bdata_obj['trans']
    obj_pose = np.concatenate([angle, trans], axis=-1)
    
    save_path_obj = save_path.replace('marker.npy', 'object.npy')
    np.save(save_path_obj, obj_pose)
    return fps


group_path = group_path
all_count = sum([len(paths) for paths in group_path])
cur_count = 0


# This will take a few hours for all datasets, here we take one dataset as an example
# 
# To accelerate the process, you could run multiple scripts like this at one time.

#
import time
for paths in group_path:
    dataset_name = paths[0].split('/')[2]
    pbar = tqdm(paths)
    pbar.set_description('Processing: %s'%dataset_name)
    fps = 0
    for path in pbar:
        save_path = path.replace('./dataset/behave/sequences_seg', save_root)
        save_path = save_path.replace('human.npz', 'marker.npy') 
        try:
            fps = behave_to_marker(path, save_path)
        except:
            print('Error: ', path)
            continue
        
    cur_count += len(paths)
    print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
    time.sleep(0.5)


