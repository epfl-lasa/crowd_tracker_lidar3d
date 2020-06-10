import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def rotate_pcl(data, quat=None): 
    if not quat: 
        # quat = [0, 0.0261769, 0, 0.9996573] # obtained from sensor calibration 
        # quat =  [0, -0.0261769, 0, 0.9996573]
        rot_mat = np.array([[  0.9961947,  0.0000000,  0.0871557],
                            [0.0000000,  1.0000000,  0.0000000],
                            [-0.0871557,  0.0000000,  0.9961947 ]])
    else: 
        r = R.from_quat(quat)
        rot_mat = r.as_dcm() # represent rotation as direct cosine matrices (3x3 real orth. mat. with determinant=1)

    data_array = data[['x', 'y', 'z']].to_numpy()
    data_rotated = np.dot(data_array, rot_mat.T)
    return data_rotated

def translate_height(data, z):
    assert data.shape[0] >= data.shape[1] # assert that columns represent features in matrix/dataframe
    if isinstance(data, pd.DataFrame):
        data = data.add(z, axis='z')
    else: 
        data[:,3] + z
    return data

def df_apply_rot(dataframe, quat=None, return_full_df=False): 
    transformed_arr = rotate_pcl(dataframe, quat)
    df_transformed = pd.DataFrame(transformed_arr, columns=['x', 'y', 'z'])
    if return_full_df: 
        dataframe['x'] = df_transformed['x']
        dataframe['y'] = df_transformed['y']
        dataframe['z'] = df_transformed['z']
        df_transformed = dataframe
    else: 
        df_transformed['intensity'] = dataframe['intensity']
    return df_transformed

def add_polar_coord(dataframe): 
    dataframe['r'] = dataframe.apply(lambda row: np.hypot(row.x, row.y), axis=1) 
    dataframe['phi'] = dataframe.apply(lambda row: np.arctan2(row.y, row.x), axis=1)
    return dataframe

def return_ground_points(dataframe,  thresh):
    df_filtered = dataframe[dataframe.z <= thresh]
    return df_filtered

def remove_ground_points(dataframe,  thresh):
    df_filtered = dataframe[dataframe.z >= thresh].reset_index(drop=True)
    return df_filtered

def standardize_data(df):
    normalized_df = df.copy()
    std_tot = np.sum([df.x.std(), df.y.std(), df.z.std()])
    for dim in ['x','y','z']:
        avg = df[dim].mean()
        normalized_df[dim] = (df[dim] - avg)/std_tot
    normalized_df['intensity'] =(df.intensity - df.intensity.mean()) / df.intensity.std()
    return normalized_df 
