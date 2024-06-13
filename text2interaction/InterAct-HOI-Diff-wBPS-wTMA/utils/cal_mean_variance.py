import numpy as np
import sys
import os
from os.path import join as pjoin


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir, markers_num):
    datasets = ['behave', 'intercap', 'neuraldome', 'chairs', 'omomo', 'imhd']
    data_list = []
    for dataset in datasets:
        dataset_dir = pjoin(data_dir, dataset, 'sequences')
        file_list = os.listdir(dataset_dir)
        for file in file_list:
            data = np.load(pjoin(dataset_dir, file, 'motion.npy'))
            if np.isnan(data).any():
                print(file)
                continue
            data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    print(Mean.shape)
    Std = data.std(axis=0)
    print(Std.shape)
    
    Std[0:markers_num*3] = Std[0:markers_num*3].mean() / 1.0
    Std[markers_num*3:markers_num*6] = Std[markers_num*3:markers_num*6].mean() / 1.0
    Std[markers_num*6:markers_num*6+7] = Std[markers_num*6:markers_num*6+7].mean() / 1.0
    Std[markers_num*6+7:markers_num*6+14] = Std[markers_num*6+7:markers_num*6+14].mean() / 1.0
    Std[markers_num*6+14:markers_num*6+20] = Std[markers_num*6+14:markers_num*6+20].mean() / 1.0
    assert markers_num*6+20 == Std.shape[-1]



    np.save(pjoin(save_dir, 'Mean_local.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_local.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    data_dir = './data'
    save_dir = './data'
    mean, std = mean_variance(data_dir, save_dir, 77)
#     print(mean)
#     print(Std)