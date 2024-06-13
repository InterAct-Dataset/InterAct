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
from human_body_prior.body_model.body_model import BodyModel
# KMP_DUPLICATE_LIB_OK=TRUE
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

dataset = sys.argv[1].upper()
with open("./data/config/{}.yml".format(dataset), 'r') as stream:
    paths = yaml.safe_load(stream)


MOTION_PATH, OBJECT_PATH, MODEL_PATH = paths['MOTION_PATH'], paths['OBJECT_TEMPLATE'], paths['MODEL_PATH']
data_name = os.listdir(MOTION_PATH)
######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl',
                        flat_hand_mean = True)

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl',
                        flat_hand_mean = True)

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl',
                        flat_hand_mean = True)

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
surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
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
    BEHAVE or InterCap for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
        
    frame_times = poses.shape[0]
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
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
    
    return verts, faces

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

    return verts, faces

# visualize surface motion of smpl model
for k, name in tqdm(enumerate(data_name)):
    print(name)
    try:
        if dataset.upper() == 'GRAB':
            verts, faces = visualize_grab(name, MOTION_PATH)
        elif dataset.upper() == 'BEHAVE':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
        elif dataset.upper() == 'NEURALDOME' or dataset.upper() == 'IMHD':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
        elif dataset.upper() == 'CHAIRS':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
        elif dataset.upper() == 'INTERCAP':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10, True)
        elif dataset.upper() == 'OMOMO':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 16)

        with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
            obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])

        mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
        obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

        angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
        obj_verts = (obj_verts)[None, ...]
        obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
        rend_video_path = os.path.join(results_folder, '{}_{}_{}.mp4'.format(dataset, name, obj_name))
        visualize_body_obj(verts, faces, obj_verts, obj_faces, save_path=rend_video_path, show_frame=True, multi_angle=True)

    except Exception as e:
        pass