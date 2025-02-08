from scipy.io import loadmat
import math
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm


def read_pose_mat(file):

    """Returns two angles pitch and yaw"""

    p = loadmat(file)['angle']
    p_x = -(p[0,0] - 0.1) + math.pi/2
    p_y = p[0,1] + math.pi/2

    return np.float32(p_x), np.float32(p_y)


def create_poses_df(path):

    """Returns a dataframe with all poses angles"""

    names = []
    p_xs = []
    p_ys = []

    df = pd.DataFrame()
    for f in tqdm(os.listdir(path)):
        p_x, p_y = read_pose_mat(file=f'{path}/{f}')
        names.append(f.split('.')[0])
        p_xs.append(p_x)
        p_ys.append(p_y)

    df['P_x'], df['P_y'], df['name'] = p_xs, p_ys, names
    return df


def sample_uniform_data(dataframe, feature_name, num_samples, num_bins, a, b, replace=True):

    """Returns a dataframe with normal distributed poses angles"""

    cages = np.linspace(a, b, num_bins+1)

    selected_data = []
    for i in range(len(cages)-1):
        cage = dataframe[(dataframe[feature_name] >= cages[i]) &
                         (dataframe[feature_name] < cages[i+1])]
        if not replace:
            try:
                selected_data.append(cage.sample(num_samples))
            except ValueError:
                selected_data.append(cage)
        else:
            selected_data.append(cage.sample(num_samples, replace=replace))

    selected_dataframe = pd.concat(selected_data)

    return selected_dataframe





