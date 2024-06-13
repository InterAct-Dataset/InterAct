import json
import os
import os.path as osp
import numpy as np

anno_path = './utils/action_label.json'

with open(anno_path, 'r') as f:
    anno = json.load(f)

id2label = {}
for des in anno['label_description']:
    id2label[des['id']] = des['description']

all_sequence = []
sequence = {}

action_label = anno['action_label']
for action in action_label:
    action_id = action['label']
    action_label = id2label[action_id]
    if action_label == 'no_interaction' or action_label == 'action_transition':
        if len(sequence) > 0:
            all_sequence.append(sequence)
            sequence = {}
        continue

    if (len(sequence) > 0) and (action_label != sequence['action']):
        all_sequence.append(sequence)
        sequence = {}

    seq_name = action['name']
    frame = int(action['frame'][1:].split('.')[0])

    if len(sequence) == 0:
        sequence = {'name': seq_name,
                    'action': action_label,
                    'frame' :[frame]}
    else:
        sequence['frame'].append(frame)

# # generate text description
# text_save_path = ./texts'
# for seq in all_sequence:
#     seq_name_fine = seq['name'] + '_{}'.format(seq['frame'][0])
#     if os.path.exists(osp.join(text_save_path, seq_name_fine) + '.txt'):
#         continue
#     print('generate text description for {}'.format(seq_name_fine))
#     obj_name = seq['name'].split('_')[2]
#     action = seq['action']
#     text = call_gpt(action, obj_name)
#     with open(osp.join(text_save_path, seq_name_fine + '.txt'), 'w') as f:
#         f.write(text)


# split behave motion annotations based on the action annotations
save_path = './dataset/behave/sequences_seg'
behave_path = './dataset/behave/sequences'
raw_behave_path = './dataset/behave-30fps-params'

for seq in all_sequence:
    seq_name_fine = seq['name'] + '_{}'.format(seq['frame'][0])
    if not osp.exists(osp.join(save_path, seq_name_fine)):
        os.makedirs(osp.join(save_path, seq_name_fine))
    min_frame = seq['frame'][0]
    max_frame = seq['frame'][-1]

    if not osp.exists(osp.join(behave_path, seq['name'], 'human.npz')):
        print('no human motion for {}'.format(seq['name']))
        continue

    # read motions
    seq_path_human = osp.join(behave_path, seq['name'], 'human.npz')
    seq_path_obj = osp.join(behave_path, seq['name'], 'object.npz')
    seq_path_human_raw = osp.join(raw_behave_path, seq['name'], 'smpl_fit_all.npz')
    seq_path_obj_raw = osp.join(raw_behave_path, seq['name'], 'object_fit_all.npz')
    seq_path_info = osp.join(raw_behave_path, seq['name'], 'info.json')
    with open(seq_path_info, "r") as f:
        seq_info = json.load(f)
    gender = seq_info["gender"]

    bdata = np.load(seq_path_human, allow_pickle=True)
    bdata_obj = np.load(seq_path_obj, allow_pickle=True)
    
    bdata_raw = np.load(seq_path_human_raw, allow_pickle=True)
    bdata_obj_raw = np.load(seq_path_obj_raw, allow_pickle=True)
    # human
    pose_h = bdata['poses']
    betas_h = bdata['betas']
    trans_h = bdata['trans']
    frame_times_h = bdata_raw['frame_times']

    # object
    angles_o = bdata_obj['angles']
    trans_o = bdata_obj['trans']
    obj_name = bdata_obj['name']
 

    pose_h_ = []
    betas_h_ = []
    trans_h_ = []
    frame_times_h_ = []

    angles_o_ = []
    trans_o_ = []
    frame_times_o_ = []

    # don't know why sometimes the object motion is smaller than human motion
    if angles_o.shape[0] < pose_h.shape[0]:
        frame_times_h = frame_times_h[:angles_o.shape[0]]
        continue
    for ind, ft in enumerate(frame_times_h):
        ft = float(ft[1:])
        if ft > (min_frame - 0.5) and ft < (max_frame + 0.5):
            pose_h_.append(pose_h[ind])
            betas_h_.append(betas_h)
            trans_h_.append(trans_h[ind])
            frame_times_h_.append(ft)

            angles_o_.append(angles_o[ind])
            trans_o_.append(trans_o[ind])
            frame_times_o_.append(ft)
    files_h = {'poses': pose_h_, 'betas': betas_h_, 'trans': trans_h_, 'frame_times': frame_times_h_, 'gender':gender}
    files_o = {'angles': angles_o_, 'trans': trans_o_, 'frame_times': frame_times_o_, 'name': obj_name}
   
    np.savez_compressed(osp.join(save_path, seq_name_fine, 'human.npz'), **files_h)
    np.savez_compressed(osp.join(save_path, seq_name_fine, 'object.npz'), **files_o)
    with open(osp.join(save_path, seq_name_fine, 'info.json'), 'w') as f:
        json.dump(seq_info, f)


# a = 1