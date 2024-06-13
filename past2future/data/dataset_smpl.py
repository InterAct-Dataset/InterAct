import json
import os
import os.path
import sys
directory = os.path.dirname(os.path.abspath(__file__))
# setting path
sys.path.append(os.path.dirname(directory))
import numpy as np
import trimesh
import torch

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import yaml
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from data.tools import vertex_normals
from data.utils import markerset_ssm67_smplh
def calculate_normal_rotation_matrix(p1, p2, p3):
    # Convert points to tensors
    p1 = torch.tensor(p1, dtype=torch.float32)
    p2 = torch.tensor(p2, dtype=torch.float32)
    p3 = torch.tensor(p3, dtype=torch.float32)
    
    # Calculate vectors v1 and v2
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Calculate the normal vector using cross product
    normal = torch.cross(v1, v2)
    
    # Normalize the normal vector
    normal_normalized = normal / torch.norm(normal)
    
    # Define the target normal (e.g., x-axis)
    target_normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    
    # Calculate rotation axis (cross product of normal and target normal)
    axis = torch.cross(normal_normalized, target_normal)
    axis_normalized = axis / torch.norm(axis)
    
    # Calculate the angle between the normal and target normal
    cos_theta = torch.dot(normal_normalized, target_normal)
    angle = torch.acos(cos_theta)
    
    # Rodrigues' rotation formula components
    K = torch.tensor([
        [0, -axis_normalized[2], axis_normalized[1]],
        [axis_normalized[2], 0, -axis_normalized[0]],
        [-axis_normalized[1], axis_normalized[0], 0]
    ], dtype=torch.float32)
    
    I = torch.eye(3)
    
    # Rotation matrix using Rodrigues' rotation formula
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.mm(K, K)
    
    return R
class Dataset(Dataset):
    def __init__(self, mode='train', past_len=10, future_len=25, sample_rate=1):
        # datasets=['./data/behave','./data/intercap','./data/neuraldome','./data/grab','./data/omomo','./data/chairs','./data/imhd']
        # datasets=['./data/behave']

        if mode == 'train':
            datasets=os.listdir('../data')
            for i in range(len(datasets)):
                datasets[i]=os.path.join('../data',datasets[i])

        elif mode == 'test':
            datasets=['../data/behave']
        else:
            raise Exception('mode must be train or test.')
        self.past_len = past_len
        self.future_len = future_len
        self.obj_dict = {}
        self.data = []
        self.idx2frame = [] # (seq_id, sub_seq_id, bias)
        cnt=0
        for dataset in datasets:
            print("loading",dataset)
            MOTION_PATH = f"{dataset}/sequences"
            OBJECT_PATH = f"{dataset}/objects"
            data_name = os.listdir(MOTION_PATH)
            dataset_size = len(data_name)
            if mode == 'train':
                data_name = data_name[:int(dataset_size * 0.9)]
            elif mode == 'test':
                data_name = data_name[int(dataset_size * 0.9):]
            else:
                raise Exception('mode must be train or test.')
            for k, name in tqdm(enumerate(data_name)):
                if (name==".DS_Store"):
                    continue
                try:
                    with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
                        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])         
                    marker_data=np.load(os.path.join(MOTION_PATH, name, 'markers.npy'), allow_pickle=True)
                except:
                    continue
                frame_times = marker_data.shape[0]
            
                if (self.obj_dict.get(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj")) is None):
                    mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
                    
                    obj_verts, obj_faces = trimesh.sample.sample_surface(mesh_obj, 128)
                    obj_verts = torch.tensor(obj_verts, dtype=torch.float32)
                    self.obj_dict[os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj")] = (obj_verts, mesh_obj.vertices, mesh_obj.faces)
                else:
                    obj_verts, _, _ = self.obj_dict[os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj")]
                records = {
                    'obj_angles': obj_angles,
                    'obj_trans': obj_trans,
                    'markers': marker_data,
                    'obj_verts': obj_verts,
                    'obj_path': os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj")
                }
                self.data.append(records)
                fragment = (past_len + future_len) * sample_rate
                for i in range(frame_times // fragment):
                    if mode == "test":
                        self.idx2frame.append((cnt, i * fragment, 1))
                    elif i == frame_times // fragment - 1:
                        self.idx2frame.append((cnt, i * fragment, frame_times + 1 - (frame_times // fragment) * fragment))
                    else:
                        self.idx2frame.append((cnt, i * fragment, fragment))
                    assert i * fragment + self.idx2frame[-1][2] < frame_times

                cnt=cnt+1
                # if (cnt==100):
                #     break
              
            
        # self.num_markers = len(markerset_ssm67_smplh)
        self.sample_rate = sample_rate
        self.num_obj_points = 128
        print("====> The number of clips for " + mode + " data: " + str(len(self.idx2frame)) + " <====")

    def __getitem__(self, idx):
        index, frame_idx, bias = self.idx2frame[idx]
        data = self.data[index]
        start_frame = np.random.choice(bias) + frame_idx
        end_frame = start_frame + (self.past_len + self.future_len) * self.sample_rate
       
        frames = []
        centroid = None
        rotation = None
        rotation_v = None
        for i in range(start_frame, end_frame, self.sample_rate):
            #smplfit_params = {'markers': data['markers'][i].copy()}
            objfit_params = {'angle': data['obj_angles'][i].copy(), 'trans': data['obj_trans'][i].copy()}
            pelvis = data['markers'][i][33].copy()
            if i == start_frame:
                centroid = pelvis
                global_orient = calculate_normal_rotation_matrix(data['markers'][i][65].copy(),data['markers'][i][3].copy(),data['markers'][i][35].copy())
                #print(global_orient)
                rotation_v = np.eye(3).astype(np.float32)
                cos, sin = global_orient[0, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2), global_orient[2, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2)
                rotation_v[[0, 2, 0, 2], [0, 2, 2, 0]] = np.array([cos, cos, -sin, sin])
                rotation = np.linalg.inv(rotation_v).astype(np.float32)

            #smplfit_params['trans'] = smplfit_params['trans'] - centroid
            pelvis = pelvis - centroid
            #pelvis_original = pelvis - smplfit_params['trans'] # pelvis position in original smpl coords system
            #smplfit_params['trans'] = np.dot(smplfit_params['trans'] + pelvis_original, rotation.T) - pelvis_original
            pelvis = np.dot(pelvis, rotation.T)
            

            markers_tran = data['markers'][i].copy()[:, :3] - centroid
            markers_tran = np.dot(markers_tran, rotation.T)
            
            markers =markers_tran

            objfit_params['trans'] = objfit_params['trans'] - centroid
            objfit_params['trans'] = np.dot(objfit_params['trans'], rotation.T)

            r_ori = Rotation.from_rotvec(objfit_params['angle'])
            r_new = Rotation.from_matrix(rotation) * r_ori
            objfit_params['angle'] = r_new.as_rotvec()

            obj_points = data['obj_verts'].clone()
            rot = r_new.as_matrix()
            obj_points[:, :3] = np.matmul(obj_points[:, :3], rot.T) + objfit_params['trans']
            # height = min(obj_points[..., 1].min(), markers[..., 1].min())
            # obj_points[..., 1] = obj_points[..., 1]height
            # markers[..., 1] = markers[..., 1] - height
            # pelvis[..., 1] = pelvis[..., 1] - height
            record = {
                'objfit_params': objfit_params,
                'obj_verts': obj_points,
                'pelvis': pelvis,
                'markers': markers
            }
            frames.append(record)

        records = {
            'frames': frames,
            'start_frame': start_frame,
            'obj_verts': data['obj_verts'],
            'obj_path': data['obj_path'],
        }
        return records

    def __len__(self):
        return len(self.idx2frame)
