# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np
import torch
import trimesh
import pyrender
import math
from render.mesh_utils import MeshViewer
from data.utils import colors,bodypart2color,marker2bodypart67_smplx
import imageio

def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:4]]
    c[3]=c[3]*255
    return c

def visualize_marker_obj(obj_verts, obj_face, pcd=None,past_len=0,  pcd_contact=None, multi_col=None, text="",
                       multi_angle=True, h=256, w=256, bg_color='white', 
                       save_path=None, fig_label=None, use_hydra_path=True, sample_rate=1,highlight_verts=None):
   
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    vis_mar = True if pcd is not None else False

    

    
    seqlen=pcd.shape[0]
    

    pcd = pcd.view(-1, 77, 3)
    pcd=np.array(pcd.cpu())

    obj_mesh_rec = obj_verts
    minx, _, miny = pcd.min(axis=(0, 1))
    maxx, _, maxy = pcd.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(pcd[:, :, 1])  # Min height

    obj_mesh_rec[:, :, 1] -= height_offset
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2
    
    pcd[:, :, 1] -= height_offset
    pcd[:, :, 0] -= (minx + maxx) / 2
    pcd[:, :, 2] -= (miny + maxy) / 2
    if isinstance(pcd, np.ndarray):
        pcd = torch.from_numpy(pcd)

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy, 
                    use_offscreen=True,
                    bg_color=bg_color)
                #ground_height=(mesh_rec.detach().cpu().numpy()[0, 0, 6633, 2] - 0.01))
    # ground plane has a bug if we want batch_size to work
    mv.render_wireframe = False
    
    if multi_angle:
        video = np.zeros([seqlen, 1 * im_height, 2 * im_width, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])
    for i in range(seqlen):
        # Rx = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        # Ry = trimesh.transformations.rotation_matrix(math.radians(180), [0, 1, 0])
        if i <= past_len:
            obj_mesh_color = np.tile(c2rgba(colors['grey']), (obj_mesh_rec.shape[1], 1))
        else:
            obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))
        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)
       
        all_meshes = []
        if vis_mar:
            m_pcd = trimesh.points.PointCloud(pcd[i])
            pcd_bodyparts = tobodyparts2(m_pcd, i <= past_len)
            for bp, m_bp in pcd_bodyparts.items():
                all_meshes.append(m_bp)
                #print(bp)

        all_meshes = all_meshes + [obj_m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()
        if multi_angle:
            video_views = [video_i]
            for _ in range(3):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                if vis_mar:
                    m_pcd.apply_transform(Ry)
                    pcd_bodyparts = tobodyparts2(m_pcd, i <= past_len)
                    for bp, m_bp in pcd_bodyparts.items():
                        all_meshes.append(m_bp)
                obj_m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                video_views.append(mv.render())
            video_i = np.concatenate((video_views[0], video_views[2]), axis=1)
        #print('video[i]',video[i].shape)
        video[i] = video_i

    if save_path is not None:
        imageio.mimsave(save_path, list(np.squeeze(video).astype(np.uint8)), fps=30 // sample_rate)
    del mv

    return

    
def tobodyparts(m_pcd, past=False):
    m_pcd = np.array(m_pcd.vertices)
    # after trnaofrming poincloud visualize for each body part separately
    pcd_bodyparts = dict()
    for bp, ids in marker2bodypart67_smplx.items():
        points = m_pcd[ids]
        #print('points',points)
        #print('ids',ids)
        tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
        tfs[:, :3, 3] = points
        col_sp = trimesh.creation.uv_sphere(radius=0.01)

        # debug markers, maybe delete it
        # if bp == 'special':
        #     col_sp = trimesh.creation.uv_sphere(radius=0.03)

        if past:
            col_sp.visual.vertex_colors = c2rgba(colors["black"])
        else:
            col_sp.visual.vertex_colors = c2rgba(colors[bodypart2color[bp]])

        pcd_bodyparts[bp] = pyrender.Mesh.from_trimesh(col_sp, poses=tfs)
    return pcd_bodyparts

def tobodyparts2(m_pcd, past=False):
    m_pcd = np.array(m_pcd.vertices)
    # after trnaofrming poincloud visualize for each body part separately
    pcd_bodyparts = dict()
    for ids in range(m_pcd.shape[0]):
        points = m_pcd[ids]
        #print('points',points)
        #print('ids',ids)
        tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
        tfs[:, :3, 3] = points
        col_sp = trimesh.creation.uv_sphere(radius=0.01)

        # debug markers, maybe delete it
        # if bp == 'special':
        #     col_sp = trimesh.creation.uv_sphere(radius=0.03)

        col_sp.visual.vertex_colors = c2rgba(colors["black"])

        pcd_bodyparts[ids] = pyrender.Mesh.from_trimesh(col_sp, poses=tfs)
    return pcd_bodyparts
