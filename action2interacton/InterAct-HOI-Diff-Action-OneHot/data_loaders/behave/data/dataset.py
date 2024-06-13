import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
from torch.utils.data._utils.collate import default_collate
from data_loaders.behave.utils.word_vectorizer import WordVectorizer
from data_loaders.behave.utils.get_opt import get_opt
import trimesh
from scipy.spatial.transform import Rotation
from data_loaders.behave.scripts.motion_process import recover_from_ric, extract_features, get_human_representation
import scipy.sparse
from data_loaders.behave.utils.paramUtil import *
from utils.utils import recover_obj_points
from data_loaders.behave.utils.plot_script import plot_3d_motion


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, datasets, split_file):
        self.mean = mean[:opt.dim_pose]
        self.std = std[:opt.dim_pose]
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 300
        self.normal_dim = opt.dim_pose

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.id_list = id_list


        
        new_name_list = []
        length_list = []
        for dataset in datasets:
            print(f"Loading dataset {dataset} ...")
            dataset_path = os.path.join(opt.data_root, dataset)
            sequences_path = os.path.join(dataset_path, 'sequences_canonical')
            sequences_names = os.listdir(sequences_path)
            for name in tqdm(sequences_names):
                try:
                    motion = np.load(pjoin(sequences_path, name, 'motion.npy'))
                    motion = motion[:300].astype(np.float32)                   
                    obj = np.load(pjoin(sequences_path, name, 'object.npz'), allow_pickle=True)
                    # load obj points----------------
                    obj_name = str(obj['name'])
                    if obj_name not in id_list:
                        # print(id_list)
                        # print(obj_name)
                        continue

                    obj_path = pjoin(dataset_path, 'objects')
                    # obj_path = pjoin(opt.data_root, 'behave', 'object_mesh')
                    # mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])

                    # temp_simp = trimesh.load(mesh_path)
                    # obj_points = np.array(temp_simp.vertices).astype(np.float32)
                    # obj_faces = np.array(temp_simp.faces).astype(np.float32)

                    # bps 
                    obj_bps = np.load(pjoin(dataset_path, 'objects_bps', obj_name, obj_name+'.npy'))
                    action = np.load(pjoin(sequences_path, name, 'action.npy'))
                    # sample object points
                    obj_sample_path = pjoin(obj_path, '{}/sample_points.npy'.format(obj_name))
                    # print(obj_sample_path)
                    # o_choose = np.load(obj_sample_path)

                    # # contact_input = np.load(pjoin(opt.data_root, 'affordance_data/contact_'+name + '.npy'), allow_pickle=True)[None][0]
                                    
                    # # center the meshes
                    # center = np.mean(obj_points, 0)
                    # obj_points -= center


                    # obj_points = obj_points[o_choose]
                    # obj_normals = obj_faces[o_choose]
                    obj_points = np.load(obj_sample_path)
                    obj_normals = np.zeros((400, 3)) 


                    data_dict[name] = { 'motion': motion,
                                        'action': action,
                                        'seq_name': dataset+'_'+obj_name,
                                        'length': len(motion),
                                        'obj_points': obj_points,
                                        'obj_normals':obj_normals,
                                        'obj_bps': obj_bps
                                        }
                    new_name_list.append(name)
                except Exception as err:
                    # print(err.__class__.__name__) # 
                    # print(err) 
                    pass
        
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        data = data.clone()
        if len(self.mean) == 559:
             data[..., :559] = data[..., :559] * self.std[:559] + self.mean[:559]
        else:
            data[..., :553] = data[..., :553] * self.std[:553] + self.mean[:553]
        return data

        
    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(
            data.device) + torch.from_numpy(self.mean).to(data.device)
        return data

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]] 

        action, seq_name, obj_points, obj_normals,obj_bps = data['action'],  data['seq_name'],  data['obj_points'], data['obj_normals'], data['obj_bps']
        length = data['length']
        motion = data['motion']
        return None, None, action, None, motion, length, None, seq_name, obj_points, obj_normals, obj_bps
        # fixed_length can be set from outside before sampling




'''For use of training text motion matching model, and evaluations, with bps and normals as input.'''
class Text2MotionDatasetV3(data.Dataset):
    def __init__(self, opt, mean, std, datasets, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 30
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40
        self.normal_dim = opt.dim_pose

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())


        new_name_list = []
        length_list = []
        for dataset in datasets:
            print(f"Loading dataset {dataset} ...")
            dataset_path = os.path.join(opt.data_root, dataset)
            sequences_path = os.path.join(dataset_path, 'sequences_canonical')
            sequences_names = os.listdir(sequences_path)
            for name in tqdm(sequences_names):
                try:
                    motion = np.load(pjoin(sequences_path, name, 'motion.npy'))
                    obj = np.load(pjoin(sequences_path, name, 'object.npz'), allow_pickle=True)
                    # load obj points----------------
                    obj_name = str(obj['name'])
                    if obj_name in id_list:
                        continue                    
                    obj_path = pjoin(dataset_path, 'objects')
                    # mesh_path = os.path.join(obj_path, obj_name, obj_name+'.obj')
                    # temp_simp = trimesh.load(mesh_path)

                    # obj_points = np.array(temp_simp.vertices)
                    # obj_faces = np.array(temp_simp.faces)

                    # center the meshes
                    # center = np.mean(obj_points, 0)
                    # obj_points -= center
                    # obj_points = obj_points.astype(np.float32)


                    # bps 
                    obj_bps = np.load(pjoin(dataset_path, 'objects_bps', obj_name, obj_name+'.npy'))
                    action = np.load(pjoin(sequences_path, name, 'action.npy'))
                    # sample object points
                    # obj_sample_path = pjoin(dataset_path, 'object_sample/{}.npy'.format(name))
                    # o_choose = np.load(obj_sample_path)
                                    
                    obj_sample_path = pjoin(obj_path, '{}/sample_points.npy'.format(obj_name))        
                    # o_choose = np.arange(200)

                    obj_points = np.load(obj_sample_path)
                    obj_normals = np.zeros((400, 3)) 



                    # TODO: hardcode
                    
                    # motion = motion[:299].astype(np.float32)


                    # contact_input = np.load(pjoin(opt.data_root, 'affordance_data/contact_'+name + '.npy'), allow_pickle=True)[None][0]

                    
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    
                    
                    data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'action': action,
                                        'seq_name': name,
                                        'obj_points': obj_points,
                                        'obj_normals':obj_normals,
                                        'obj_bps': obj_bps,
                                        # 'gt_afford_labels':contact_input
                                    }

                    new_name_list.append(name)
                    length_list.append(len(motion))
                except Exception as err:
                    print(err.__class__.__name__) 
                    print(err)
                    print(name) 
                    pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        data = data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
        return data

    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(
            data.device) + torch.from_numpy(self.mean).to(data.device)
        return data


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, action, seq_name, obj_points, obj_normals, obj_bps = data['motion'], data['length'], data['action'], data['seq_name'],  data['obj_points'], data['obj_normals'], data['obj_bps']



        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        
        if not self.opt.use_global:
            "Z Normalization"
            motion = np.copy(motion)
            if len(self.mean) == 559:
                motion[:,:559] = (motion[:, :559] - self.mean[:559]) / self.std[:559]
            else:
                #  for evaluation of ground truth
                motion[..., :553] = (motion[..., :553] - self.mean[:553]) / self.std[:553]

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)


        # Contact labels here for evaluation!
        return None, None, action, None, motion, m_length, None, seq_name, obj_points, obj_normals, obj_bps



# A wrapper class for behave dataset t2m and t2afford
class Behave(data.Dataset):
    def __init__(self, mode, 
                    datapath='./data/behave_opt.txt', 
                    split="train",
                    dataset = 'interact',
                    use_global=False,
                    training_stage=1,
                    wo_obj_motion=False,
                    **kwargs):
        self.mode = mode


        self.dataset_name = 't2m_behave'
        self.dataname = 't2m_behave'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device, use_global, wo_obj_motion)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './data'
        if dataset == 'interact':
            self.datasets = ['behave', 'intercap', 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab']
        else:
            self.datasets = [dataset]
        self.opt = opt
        self.use_global = use_global
        self.training_stage = training_stage
        print('Loading dataset %s ...' % self.datasets)

        if  self.training_stage==1:
            self.split_file = pjoin(opt.meta_dir, f'{split}.txt')     #   adopt augmented data for affordance training
            if mode == 'text_only':
                self.t2m_dataset = TextOnlyAffordDataset(self.opt, self.split_file)
            else:
                self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
                self.t2m_dataset = Text2AffordDataset(self.opt,  self.split_file, self.w_vectorizer)

        elif  self.training_stage>=2:

            # used by our models
            self.mean = np.load(pjoin(opt.meta_dir, 'Mean_all_can.npy'))
            self.std = np.load(pjoin(opt.meta_dir, 'Std_all_can.npy'))

  

            self.split_file = pjoin(opt.meta_dir, 'test.txt')
            if mode == 'text_only':
                self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.datasets, self.split_file)
            else:
                self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
                self.t2m_dataset = Text2MotionDatasetV3(self.opt, self.mean, self.std, self.datasets, self.split_file, self.w_vectorizer)
                self.num_actions = 16 # dummy placeholder

        else:
            print(f"error!")
        # print(len(self.t2m_dataset))
        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'


        # Load necessay variables for converting raw motion to processed data


        # # Get offsets of target skeleton
        # example_data = np.load(data_dir)
        # example_data = example_data.reshape(len(example_data), -1, 3)
        # example_data = torch.from_numpy(example_data)
        # tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
        # # (joints_num, 3)
        # tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


    def motion_to_rel_data(self, motion, model, is_norm=False):

        motion_bu = motion.detach().clone()
        # Right/Left foot
        fid_r, fid_l = [61, 52, 53, 40, 34, 49, 40], [29, 30, 18, 19, 7, 2, 15]
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        
        sample_rel_np_list = []
        for ii in range(len(motion)):
            # Data need to be [120 (timestep), 22, 3] to get feature
            sample_rel = get_human_representation(
                motion[ii].detach().cpu().clone().permute(2, 0,
                                                          1).cpu().numpy(),
                0.002, fid_r, fid_l)
            # Duplicate last motion step to match the size
            sample_rel = torch.from_numpy(sample_rel).unsqueeze(0).float()
            # sample_rel = torch.cat(
            #     [sample_rel, sample_rel[0:1, -1:, :].clone()], dim=1)
            
            # Normalize with relative normalization
            if is_norm:
                sample_rel = (sample_rel - self.mean_rel[:553]) / self.std_rel[:553]
            sample_rel = sample_rel.unsqueeze(1).permute(0, 3, 1, 2)
            sample_rel = sample_rel.to(motion.device)
            sample_rel_np_list.append(sample_rel)

        processed_data = torch.cat(sample_rel_np_list, axis=0)



        n_markers = 77
        # NOTE: check if the sequence is still that same after extract_features and converting back
        # sample = dataset.t2m_dataset.inv_transform(sample_abs.cpu().permute(0, 2, 3, 1)).float()
        # sample_after = (processed_data.permute(0, 2, 3, 1) * self.std_rel) + self.mean_rel
        
        
        # print(f"processed_data:{processed_data.shape}  {sample_after.shape}")
        # B, _, T , F = sample_after.shape
        # sample_after = sample_after[..., :66].reshape(B, T, n_joints, 3).permute(0,2,3,1)

        # sample_after = recover_from_ric(sample_after, n_joints)
        # sample_after = sample_after.view(-1, *sample_after.shape[2:]).permute(0, 2, 3, 1)

        # rot2xyz_pose_rep = 'xyz'
        # rot2xyz_mask = None
        # sample_after = model.rot2xyz(x=sample_after,
        #                     mask=rot2xyz_mask,
        #                     pose_rep=rot2xyz_pose_rep,
        #                     glob=True,
        #                     translation=True,
        #                     jointstype='smpl',
        #                     vertstrans=True,
        #                     betas=None,
        #                     beta=0,
        #                     glob_rot=None,
        #                     get_rotations_back=False)

        # from data_loaders.humanml.utils.plot_script import plot_3d_motion


        # for i in range(motion.shape[0]):
        #     # print(f"test:{ sample_after.shape}   {motion[2].permute(2,0,1).shape}")
        #     plot_3d_motion("./test_positions_{}.mp4".format(i), self.kinematic_chain, motion[i].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)
        #     plot_3d_motion("./test_positions_1_after{}.mp4".format(i), self.kinematic_chain, sample_after[i].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

        # Return data already normalized with relative mean and std. shape [bs, 553, 1, 120(motion step)]
        return processed_data



