# NOTE: Canonicalize the first human pose

import json
import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
import yaml
import smplx
import trimesh
from scipy.spatial.transform import Rotation
from copy import copy
from render.mesh_viz import visualize_body_obj
from visualize_marker import plot_markers 
from utils.markerset import *
import sys
import shutil
from human_body_prior.body_model.body_model import BodyModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


MODEL_PATH = '../data/models'

######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl',
                        flat_hand_mean=True)

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl',
                        flat_hand_mean=True)

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl',
                        flat_hand_mean=True)

smplh10 = {'male': smplh_model_male, 'female': smplh_model_female, 'neutral': smplh_model_neutral}
######################################## smplx 10 ########################################
smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                        gender = 'male',
                        use_pca=False,
                        ext='pkl')
                           
smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                        gender="female",
                        use_pca=False,
                        ext='pkl')

smplx_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl')

smplx10 = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smplx_model_neutral}
######################################## smplx 10 pca 12 ########################################
smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="male",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')

smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="female",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')
smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="neutral",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')
smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}
######################################## smplh 16 ########################################
SMPLH_PATH = MODEL_PATH+'/smplh'
surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
surface_model_female_fname = os.path.join(SMPLH_PATH, "male", "model.npz")
surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
dmpl_fname = None
num_dmpls = None 
num_expressions = None
num_betas = 16 

smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16 = {'male': smplh16_model_male, 'female': smplh16_model_female, 'neutral': smplh16_model_neutral}
######################################## smplx 16 ########################################
SMPLX_PATH = MODEL_PATH+'/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16 = {'male': smplx16_model_male, 'female': smplx16_model_female, 'neutral': smplx16_model_neutral}
########################################################################################
results_folder = "./results"
os.makedirs(results_folder, exist_ok=True)

######################################## Visualize SMPL ########################################
def visualize_smpl(name, MOTION_PATH, model_type, num_betas, use_pca=False):
    """
    BEHAVE for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    InterCap for SMPLX 10 PCA 12
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    
    
    frame_times = poses.shape[0]
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            if len(betas) == len(poses):
                betas = betas[0]
            smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                global_orient=torch.from_numpy(poses[:, :3]).float(),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                transl=torch.from_numpy(trans).float(),) 
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:78]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 78:90]).float(),
                              jaw_pose=torch.zeros(frame_times, 3).float(),
                              leye_pose=torch.zeros(frame_times, 3).float(),
                              reye_pose=torch.zeros(frame_times, 3).float(),
                              expression=torch.zeros(frame_times, 10).float(),
                              betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                              transl=torch.from_numpy(trans).float(),)
            else:
                smpl_model = smplx10[gender]
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                    global_orient=torch.from_numpy(poses[:, :3]).float(),
                                    left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                    right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                    jaw_pose = torch.zeros([frame_times,3]).float(),
                                    reye_pose = torch.zeros([frame_times,3]).float(),
                                    leye_pose = torch.zeros([frame_times,3]).float(),
                                    expression = torch.zeros([frame_times,10]).float(),
                                    betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                    transl=torch.from_numpy(trans).float(),)
        verts = to_cpu(smplx_output.vertices)
        faces = smpl_model.faces
        joints = to_cpu(smplx_output.joints)
    elif num_betas == 16: 
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
                            pose_hand=torch.from_numpy(poses[:, 66:156]).float(), 
                            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
                            root_orient=torch.from_numpy(poses[:, :3]).float(), 
                            trans=torch.from_numpy(trans).float())
        verts = to_cpu(smplx_output.v)
        faces = smpl_model.f
        joints = to_cpu(smplx_output.Jtr)
    
    return verts, faces, joints

######################################## utils for GRAB ########################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def points2sphere(points, radius = .001, vc = [0., 0., 1.], count = [5,5]):

    points = points.reshape(-1,3)
    n_points = points.shape[0]

    spheres = []
    for p in range(n_points):
        sphs = trimesh.creation.uv_sphere(radius=radius, count = count)
        sphs.apply_translation(points[p])
        sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)

        spheres.append(sphs)

    spheres = Mesh.concatenate_meshes(spheres)
    return spheres

class Mesh(trimesh.Trimesh):

    def __init__(self,
                 filename=None,
                 vertices=None,
                 faces=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 process = False,
                 visual = None,
                 wireframe=False,
                 smooth = False,
                 **kwargs):

        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process = process)
            vertices = mesh.vertices
            faces= mesh.faces
            visual = mesh.visual
        if vscale is not None:
            vertices = vertices*vscale

        if faces is None:
            mesh = points2sphere(vertices)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self,color, array, ids):

        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vertex_colors(self,vc, vertex_ids = None):

        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self,fc, face_ids = None):

        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def concatenate_meshes(meshes):
        return trimesh.util.concatenate(meshes)
def DotDict(in_dict):
    
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}

sbj_info = {}
def load_sbj_verts(sbj_id, seq_data, data_root_folder = './data/grab/'):
    
    mesh_path = os.path.join(data_root_folder,seq_data.body.vtemp)
    if sbj_id in sbj_info:
        sbj_vtemp = sbj_info[sbj_id]
    else:
        sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
        sbj_info[sbj_id] = sbj_vtemp
    return sbj_vtemp
######################################## Visualize GRAB ########################################
def visualize_grab(name, MOTION_PATH):
    """
    vertices: (N, 10475, 3)
    """
    motion_file = os.path.join(MOTION_PATH,name,'motion.npz')
    seq_data = parse_npz(motion_file)
    n_comps = seq_data['n_comps']
    gender = seq_data['gender']
    sbj_id = seq_data['sbj_id']
    T = seq_data.n_frames
    sbj_vtemp = load_sbj_verts(sbj_id, seq_data, os.path.dirname(MOTION_PATH))

    smpl_model = smplx.create( 
        model_path=MODEL_PATH,
        model_type='smplx',
        gender=gender,
        num_pca_comps=n_comps,
        v_template = sbj_vtemp,
        batch_size=T).cuda()
    sbj_parms = params2torch(seq_data.body.params)

    smplx_output = smpl_model(**sbj_parms)
    verts = to_cpu(smplx_output.vertices)
    faces = smpl_model.faces
    joints = to_cpu(smplx_output.joints)

    return verts, faces, joints



if __name__ == "__main__":
    datasets = ['behave', 'intercap', 'neuraldome', 'grab', 'chairs', 'omomo', 'imhd']
    data_root = '../data'
    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset)
        MOTION_PATH = os.path.join(dataset_path, 'sequences')
        NEW_MOTION_PATH = os.path.join(dataset_path, 'sequences_canonical')
        OBJECT_PATH = os.path.join(dataset_path, 'objects')
        data_name = os.listdir(MOTION_PATH)
        for name in data_name:
            try :
                if dataset.upper() == 'GRAB':
                    verts, faces, joints = visualize_grab(name, MOTION_PATH)
                    markers = verts[:,markerset_smplx]
                elif dataset.upper() == 'BEHAVE':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
                    markers = verts[:,markerset_smplh]
                elif dataset.upper() == 'NEURALDOME' or dataset.upper() == 'IMHD':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
                    markers = verts[:,markerset_smplh]
                elif dataset.upper() == 'CHAIRS':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
                    markers = verts[:,markerset_smplx]
                elif dataset.upper() == 'INTERCAP':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplx', 10, True)
                    markers = verts[:,markerset_smplx]
                elif dataset.upper() == 'OMOMO':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplx', 16)
                    markers = verts[:,markerset_smplx]
                # np.save(os.path.join(MOTION_PATH, name, 'markers.npy'), markers)
                centroid = joints[0,0]
            
                with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
                    obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
                with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
                    poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
                global_orient = Rotation.from_rotvec(poses[0,:3]).as_matrix()
                rotation_v = np.eye(3).astype(np.float32)
                cos, sin = global_orient[0, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2), global_orient[2, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2)
                rotation_v[[0, 2, 0, 2], [0, 2, 2, 0]] = np.array([cos, cos, -sin, sin])
                rotation = np.linalg.inv(rotation_v).astype(np.float32)

                new_poses = []
                new_trans = []
                new_obj_angles = []
                new_obj_trans = []
                for i in range(poses.shape[0]):
                    smplfit_params = {'pose': poses[i].copy(), 'trans': trans[i].copy()}
                    objfit_params = {'angle': obj_angles[i].copy(), 'trans': obj_trans[i].copy()}                
                    pelvis = joints[i,0]
                    smplfit_params['trans'] = smplfit_params['trans'] - centroid
                    pelvis = pelvis - centroid
                    pelvis_original = pelvis - smplfit_params['trans'] # pelvis position in original smpl coords system
                    smplfit_params['trans'] = np.dot(smplfit_params['trans'] + pelvis_original, rotation.T) - pelvis_original
                    pelvis = np.dot(pelvis, rotation.T)
                    # smpl pose parameter in the canonical system
                    r_ori = Rotation.from_rotvec(smplfit_params['pose'][:3])
                    r_new = Rotation.from_matrix(rotation) * r_ori
                    smplfit_params['pose'][:3] = r_new.as_rotvec()
                    # object in the canonical system
                    objfit_params['trans'] = objfit_params['trans'] - centroid
                    objfit_params['trans'] = np.dot(objfit_params['trans'], rotation.T)
                    r_ori = Rotation.from_rotvec(objfit_params['angle'])
                    r_new = Rotation.from_matrix(rotation) * r_ori
                    objfit_params['angle'] = r_new.as_rotvec()
                    new_poses.append(smplfit_params['pose'])
                    new_trans.append(smplfit_params['trans'])
                    new_obj_angles.append(objfit_params['angle'])
                    new_obj_trans.append(objfit_params['trans'])
                
                new_human = {
                    'poses': np.array(new_poses),
                    'betas': betas,
                    'trans': np.array(new_trans),
                    'gender': gender
                }
                os.makedirs(os.path.join(NEW_MOTION_PATH, name), exist_ok=True)
                np.savez(os.path.join(NEW_MOTION_PATH, name, 'human.npz'), **new_human)

                # get smpl vertices and object vertices
                if dataset.upper() == 'GRAB':
                    verts, faces, joints = visualize_grab(name, NEW_MOTION_PATH)
                elif dataset.upper() == 'BEHAVE':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplh', 10)
                elif dataset.upper() == 'NEURALDOME' or dataset.upper() == 'IMHD':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplh', 16)
                elif dataset.upper() == 'CHAIRS':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplx', 10)
                elif dataset.upper() == 'INTERCAP':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplx', 10, True)
                elif dataset.upper() == 'OMOMO':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplx', 16)
                
                mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
                obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
                new_obj_trans = np.array(new_obj_trans)
                new_obj_angles = np.array(new_obj_angles)
                angle_matrix = Rotation.from_rotvec(new_obj_angles).as_matrix()
                obj_verts = mesh_obj.vertices[None, ...]
                obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix[:30], (0, 2, 1))) + new_obj_trans[:30, None, :]

                
                diff_fix = min(verts[:30, ..., 1].min(), obj_verts[:30, ..., 1].min())
                new_trans = np.array(new_trans)
                new_obj_trans[..., 1] -= diff_fix
                new_trans[..., 1] -= diff_fix

                new_human = {
                    'poses': np.array(new_poses),
                    'betas': betas[0],
                    'trans': np.array(new_trans),
                    'gender': gender
                }
                new_obj = {
                    'angles': np.array(new_obj_angles),
                    'trans': np.array(new_obj_trans),
                    'name': obj_name
                }


                np.savez(os.path.join(NEW_MOTION_PATH, name, 'human.npz'), **new_human)
                np.savez(os.path.join(NEW_MOTION_PATH, name, 'object.npz'), **new_obj)
                if os.path.exists(os.path.join(MOTION_PATH, name, 'text.txt')) and os.path.isdir(os.path.join(NEW_MOTION_PATH, name)):
                    shutil.copy(os.path.join(MOTION_PATH, name, 'text.txt'), os.path.join(NEW_MOTION_PATH, name, 'text.txt'))
                if os.path.exists(os.path.join(MOTION_PATH, name, 'action.txt')) and os.path.isdir(os.path.join(NEW_MOTION_PATH, name)):
                    shutil.copy(os.path.join(MOTION_PATH, name, 'action.txt'), os.path.join(NEW_MOTION_PATH, name, 'action.txt'))
                if os.path.exists(os.path.join(MOTION_PATH, name, 'action.npy')) and os.path.isdir(os.path.join(NEW_MOTION_PATH, name)):
                    shutil.copy(os.path.join(MOTION_PATH, name, 'action.npy'), os.path.join(NEW_MOTION_PATH, name, 'action.npy'))
            except Exception as e:
                print(e)
                print(name)

                
            




                
