import sys
sys.path.append("../../")

import os
import numpy as np
import joblib 
import json 
import trimesh 
import time 

import torch
from torch.utils.data import Dataset

import pytorch3d.transforms as transforms 

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from bps_torch.tools import sample_uniform_cylinder
from scipy.spatial.transform import Rotation
from human_body_prior.body_model.body_model import BodyModel

from manip.lafan1.utils import rotate_at_frame_w_obj 


MODEL_PATH = "../data/models/"
SMPLH_PATH = MODEL_PATH + "smplh/"

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def rotate(points, R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1,2))
    return r_points.reshape(shape)

def get_smpl_parents(use_joints24=False):
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents() 

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3 
    parents = get_smpl_parents() 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat 

def quat_fk_torch(lrot_mat, lpos, use_joints24=False):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    if use_joints24:
        parents = get_smpl_parents(use_joints24=True)
    else:
        parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        if i < lrot.shape[-2]:
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

def merge_two_parts(verts_list, faces_list):
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        # part_verts = torch.from_numpy(verts_list[p_idx]) # T X Nv X 3 
        part_verts = verts_list[p_idx] # T X Nv X 3 
        part_faces = torch.from_numpy(faces_list[p_idx]) # T X Nf X 3 

        if p_idx == 0:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces)
        else:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces+verts_num)

        verts_num += part_verts.shape[1] 

    # merged_verts = torch.cat(merged_verts_list, dim=1).data.cpu().numpy()
    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).data.cpu().numpy() 

    return merged_verts, merged_faces 

    
class MarkerManipDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        use_object_splits=True,
        use_all_data=True
    ):
        self.train = train
        
        self.window = window

        self.use_joints24 = True 

        self.use_object_splits = True 
        # self.train_objects = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", \
        #             "trashcan", "monitor", "floorlamp", "clothesstand", "vacuum"] # 10 objects 
        self.test_objects = ["chairblack", "chairwood", "chair", "woodchair", "whitechair", 
                            "table", "talltable", "desk", "cylinderlarge", "pyramidlarge",
                            "spherelarge", "toruslarge", "15", '17', '24']

        self.data_root_folder = data_root_folder 
        
        self.window_data_dict = {}
        self.data_dict = {}

        self.bps_path = "./manip/data/bps.pt"

        min_max_mean_std_data_path = os.path.join(data_root_folder, "min_max_mean_std_data_window_"+str(self.window)+".p")

        if use_all_data:
            self.datasets = ['behave', 'intercap', 'neuraldome', 'grab', 'chairs', 'omomo', 'imhd']
        else:
            self.datasets = ['omomo']

        dataset_path = os.path.join(data_root_folder, "neuraldome")

        seq_data_path = os.path.join(dataset_path, "sequences_canonical")
        processed_data_path = os.path.join(dataset_path, "sequences_canonical_window")

        
        self.prep_bps_data()

        if os.path.exists(processed_data_path):
            self.load_window_data_dict()


        else:
            if os.path.exists(seq_data_path):
                self.load_data_dict()
        
            self.cal_normalize_data_input()           

        # # Mannually enable this.
        # self.get_bps_from_window_data_dict()
        
        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            if self.train:
                min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(min_max_mean_std_jpos_data, min_max_mean_std_data_path)
           
        self.global_markers_min = torch.from_numpy(min_max_mean_std_jpos_data['global_markers_min']).float().reshape(77, 3)[None]
        self.global_markers_max = torch.from_numpy(min_max_mean_std_jpos_data['global_markers_max']).float().reshape(77, 3)[None]
       
        if self.use_object_splits:
            self.window_data_dict = self.filter_out_object_split()

        # Get train and validation statistics. 
        if self.train:
            print("Total number of windows for training:{0}".format(len(self.window_data_dict)))
        else:
            print("Total number of windows for validation:{0}".format(len(self.window_data_dict)))

        # # Prepare SMPLX model 
        # soma_work_base_dir = os.path.join(self.data_root_folder, 'smpl_all_models')
        # support_base_dir = soma_work_base_dir 
        # surface_model_type = "smplx"
        # # surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "male", 'model.npz')
        # # surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "female", 'model.npz')
        # surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_MALE.npz")
        # surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_FEMALE.npz")
        # dmpl_fname = None
        # num_dmpls = None 
        # num_expressions = None
        # num_betas = 16 

        # self.male_bm = BodyModel(bm_fname=surface_model_male_fname,
        #                 num_betas=num_betas,
        #                 num_expressions=num_expressions,
        #                 num_dmpls=num_dmpls,
        #                 dmpl_fname=dmpl_fname)
        # self.female_bm = BodyModel(bm_fname=surface_model_female_fname,
        #                 num_betas=num_betas,
        #                 num_expressions=num_expressions,
        #                 num_dmpls=num_dmpls,
        #                 dmpl_fname=dmpl_fname)

        # for p in self.male_bm.parameters():
        #     p.requires_grad = False
        # for p in self.female_bm.parameters():
        #     p.requires_grad = False 

        # self.male_bm = self.male_bm.cuda()
        # self.female_bm = self.female_bm.cuda()
        
        # self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}
    def __len__(self):
        return len(self.window_data_dict)
    def __getitem__(self, index):
        data_input = self.window_data_dict[index]['motion']
        markers = torch.from_numpy(self.window_data_dict[index]['markers']).float()
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.window_data_dict[index]['seq_name'] 
        obj_name = self.window_data_dict[index]['obj_name']


        dataset = self.window_data_dict[index]['dataset']

        dest_obj_bps_npy_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name,"object_bps.npy")

        obj_bps_data = np.load(dest_obj_bps_npy_path) # T X N X 3 
        obj_bps_data = torch.from_numpy(obj_bps_data) 
        num_markers = 77
        ori_data_input = data_input.clone()
        normalized_markers = self.normalize_markers_min_max(markers) # T X 77 X 3
        new_data_input =  torch.cat((normalized_markers.reshape(-1, num_markers*3), data_input[:,num_markers*3:]), dim=1)
        

        # Add padding. 
        actual_steps = new_data_input.shape[0]
        if actual_steps < self.window:
            paded_new_data_input = torch.cat((new_data_input, torch.zeros(self.window-actual_steps, new_data_input.shape[-1])), dim=0)
            paded_ori_data_input = torch.cat((ori_data_input, torch.zeros(self.window-actual_steps, ori_data_input.shape[-1])), dim=0)  
            paded_markers = torch.cat((markers, torch.zeros(self.window-actual_steps, markers.shape[1], markers.shape[2])), dim=0)  
            obj_bps_data = obj_bps_data[:actual_steps]
            paded_obj_bps = torch.cat((obj_bps_data.reshape(actual_steps, -1), \
                torch.zeros(self.window-actual_steps, obj_bps_data.reshape(actual_steps, -1).shape[1])), dim=0)
        
            paded_obj_angles = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_angles']).float(), \
                torch.zeros(self.window-actual_steps, 3)), dim=0)
            paded_obj_trans = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_trans']).float(), \
                torch.zeros(self.window-actual_steps, 3)), dim=0)

        else:
            paded_new_data_input = new_data_input 
            paded_ori_data_input = ori_data_input 
            paded_markers = markers
            paded_obj_bps = obj_bps_data.reshape(new_data_input.shape[0], -1)  
            paded_obj_angles = torch.from_numpy(self.window_data_dict[index]['obj_angles']).float()
            paded_obj_trans = torch.from_numpy(self.window_data_dict[index]['obj_trans']).float()
        data_input_dict = {}
        data_input_dict['motion'] = paded_new_data_input
        data_input_dict['ori_motion'] = paded_ori_data_input 
        data_input_dict['markers'] = paded_markers
        data_input_dict['obj_bps'] = paded_obj_bps

        data_input_dict['obj_angles'] = paded_obj_angles
        data_input_dict['obj_trans'] = paded_obj_trans
        data_input_dict['dataset'] = dataset 

        data_input_dict['seq_name'] = seq_name
        data_input_dict['obj_name'] = obj_name

        data_input_dict['seq_len'] = actual_steps 

        return data_input_dict

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0 
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(1, -1, 3)
            
            bps = {
                'obj': bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            torch.save(bps, self.bps_path)
        
        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps['obj'].cuda()    

    def load_window_data_dict(self):
        self.window_data_dict = {}
        s_idx = 0
        for dataset in self.datasets:
            dataset_path = os.path.join(self.data_root_folder, dataset)
            seq_data_path = os.path.join(dataset_path, "sequences_canonical_window")
            for seq in os.listdir(seq_data_path):
                seq_path = os.path.join(seq_data_path, seq)
                object_data_path = os.path.join(seq_path, "object.npz")
                object_data = np.load(object_data_path, allow_pickle=True)
                motion_path = os.path.join(seq_path, "motion.npy")
                if not os.path.exists(motion_path):
                    continue
                motion = np.load(motion_path)
                length = len(motion) 
                if length < 60:
                    continue
                self.window_data_dict[s_idx] = {}
                self.window_data_dict[s_idx]['obj_name'] = str(object_data['name'])
                self.window_data_dict[s_idx]['dataset'] = dataset
                self.window_data_dict[s_idx]['obj_trans'] = object_data['trans']
                self.window_data_dict[s_idx]['obj_angles'] = object_data['angles']
                self.window_data_dict[s_idx]['markers'] = np.load(os.path.join(seq_path, "markers.npy"))
                self.window_data_dict[s_idx]['motion'] = motion
                self.window_data_dict[s_idx]['seq_name'] = seq
                self.window_data_dict[s_idx]['seq_len'] = length 
                s_idx += 1
            
    
    def load_data_dict(self):
        self.data_dict = {}
        s_idx = 0
        for dataset in self.datasets:
            dataset_path = os.path.join(self.data_root_folder, dataset)
            seq_data_path = os.path.join(dataset_path, "sequences_canonical")
            for seq in os.listdir(seq_data_path):
                self.data_dict[s_idx] = {}
                seq_path = os.path.join(seq_data_path, seq)
                object_data_path = os.path.join(seq_path, "object.npz")
                object_data = np.load(object_data_path, allow_pickle=True)
                self.data_dict[s_idx]['obj_name'] = str(object_data['name'])
                self.data_dict[s_idx]['obj_trans'] = object_data['trans']
                self.data_dict[s_idx]['obj_angles'] = object_data['angles']
                self.data_dict[s_idx]['markers'] = np.load(os.path.join(seq_path, "markers.npy"))
                self.data_dict[s_idx]['motion'] = np.load(os.path.join(seq_path, "motion.npy"))
                self.data_dict[s_idx]['seq_name'] = seq
                self.data_dict[s_idx]['seq_len'] = len(self.data_dict[s_idx]['motion']) 
                self.data_dict[s_idx]['dataset'] = dataset
                s_idx += 1
            
    
    def cal_normalize_data_input(self):
        self.window_data_dict = {}
        s_idx = 0
        for index in self.data_dict:
            seq_name = self.data_dict[index]['seq_name']

            object_name = self.data_dict[index]['obj_name']

            markers = self.data_dict[index]['markers'] # T X 77 X 3

            motion = self.data_dict[index]['motion'] #T X 559
            
            obj_trans = self.data_dict[index]['obj_trans'] # T X 3
            obj_angles = self.data_dict[index]['obj_angles'] # T X 3 
            dataset = self.data_dict[index]['dataset']
            dataset_path = os.path.join(self.data_root_folder, dataset)
            seq_data_path = os.path.join(dataset_path, "sequences_canonical_window")
            os.makedirs(seq_data_path, exist_ok=True)

           
            num_steps = self.data_dict[index]['seq_len']
            for start_t_idx in range(0, num_steps, self.window//2):
                end_t_idx = start_t_idx + self.window 
                if end_t_idx >= num_steps:
                    end_t_idx = num_steps 

                # Skip the segment that has a length < 60 
                if end_t_idx - start_t_idx < 60:
                    continue 

                self.window_data_dict[s_idx] = {}
                self.window_data_dict[s_idx]['motion'] = motion[start_t_idx:end_t_idx,:553]
                self.window_data_dict[s_idx]['seq_name'] = seq_name + '_' + str(start_t_idx)
                self.window_data_dict[s_idx]['obj_name'] = object_name
                self.window_data_dict[s_idx]['markers'] = markers[start_t_idx:end_t_idx]
                self.window_data_dict[s_idx]['obj_trans'] = obj_trans[start_t_idx:end_t_idx]
                self.window_data_dict[s_idx]['obj_angles'] = obj_angles[start_t_idx:end_t_idx]
                self.window_data_dict[s_idx]['seq_len'] = end_t_idx - start_t_idx 
                self.window_data_dict[s_idx]['dataset'] = dataset

                obj = {
                    'angles': self.window_data_dict[s_idx]['obj_angles'],
                    'trans': self.window_data_dict[s_idx]['obj_trans'],
                    'name': self.window_data_dict[s_idx]['obj_name']
                }
                save_path = os.path.join(seq_data_path, self.window_data_dict[s_idx]['seq_name'])
                os.makedirs(save_path, exist_ok=True)
                np.savez(os.path.join(save_path,'object.npz'), **obj)
                np.save(os.path.join(save_path, 'markers.npy'), self.window_data_dict[s_idx]['markers'])
                np.save(os.path.join(save_path, 'motion.npy'), self.window_data_dict[s_idx]['motion'])

                s_idx += 1

    def get_bps_from_window_data_dict(self):
        # Given window_data_dict which contains canonizalized information, compute its corresponding BPS representation. 
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]

            seq_name = window_data['seq_name']
            obj_name = window_data['obj_name']
            obj_trans = window_data['obj_trans']
            obj_angles = window_data['obj_angles']
            dataset = window_data['dataset']
            seq_data_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name)
            obj_mesh_path = os.path.join(self.data_root_folder, dataset, "objects", f"{obj_name}/{obj_name}.obj")
            mesh_obj = trimesh.load(obj_mesh_path)
            obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

            angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
            obj_verts = (obj_verts)[None, ...]
            obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

            torch_obj_verts = torch.from_numpy(obj_verts).float().cuda()
            torch_obj_trans = torch.from_numpy(obj_trans).float().cuda()
            dest_obj_bps_npy_path = os.path.join(seq_data_path, "object_bps.npy")

            # Get object geometry 

            if not os.path.exists(dest_obj_bps_npy_path):
                object_bps = self.compute_object_geo_bps(torch_obj_verts, torch_obj_trans)
                np.save(dest_obj_bps_npy_path, object_bps.data.detach().cpu().numpy()) 

        import pdb 
        pdb.set_trace() 

    def compute_object_geo_bps(self, obj_verts, obj_trans):
    # obj_verts: T X Nv X 3, obj_trans: T X 3
        bps_object_geo = self.bps_torch.encode(x=obj_verts, \
                    feature_type=['deltas'], \
                    custom_basis=self.obj_bps.repeat(obj_trans.shape[0], \
                    1, 1)+obj_trans[:, None, :])['deltas'] # T X N X 3 

        return bps_object_geo
    
    def filter_out_object_split(self):
        # Remove some sequences from window_data_dict such that we have some unseen objects during testing. 
        new_cnt = 0
        new_window_data_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            object_name = window_data['obj_name']
            if self.train and object_name not in self.test_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_cnt += 1

            if (not self.train) and object_name in self.test_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_cnt += 1

        return new_window_data_dict
    
    def extract_min_max_mean_std_from_data(self):
        all_markers_data = []

        for s_idx in self.window_data_dict:
            curr_window_data = self.window_data_dict[s_idx]['motion'] # T X D 

            all_markers_data.append(curr_window_data[:, :77*3])


            curr_seq_name = self.window_data_dict[s_idx]['seq_name']

        all_markers_data = np.vstack(all_markers_data).reshape(-1, 231) # (N*T) X 77*3 
  

        min_markers = all_markers_data.min(axis=0)
        max_markers = all_markers_data.max(axis=0)


        stats_dict = {}
        stats_dict['global_markers_min'] = min_markers 
        stats_dict['global_markers_max'] = max_markers
   

        return stats_dict 
    
    def normalize_markers_min_max(self, ori_makers):
        # ori_makers: T X 77 X 3 
        normalized_markers = (ori_makers - self.global_markers_min.to(ori_makers.device))/(self.global_markers_max.to(ori_makers.device)\
        -self.global_markers_min.to(ori_makers.device))
        normalized_markers = normalized_markers * 2 - 1 # [-1, 1] range 

        return normalized_markers # T X 77 X 3 
    
    
    def de_normalize_markers_min_max(self, normalize_markers):
        normalize_markers = (normalize_markers + 1) * 0.5 # [0, 1] range
        de_markers = normalize_markers * (self.global_markers_max.to(normalize_markers.device)-\
        self.global_markers_min.to(normalize_markers.device)) + self.global_markers_min.to(normalize_markers.device)

        return de_markers # T X 77 X 3 