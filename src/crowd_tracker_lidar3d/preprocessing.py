import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def rotate_pcl(data, quat=None): 
    if not quat: 
        quat = [0, 0.0261769, 0, 0.9996573] # obtained from sensor calibration 
    r = R.from_quat(quat)
    rot_mat = r.as_dcm() # represent rotation as direct cosine matrices (3x3 real orth. mat. with determinant=1)

        # R = [  0.9975641,  0.0000000,  0.0697565;
        #        0.0000000,  1.0000000,  0.0000000;
        #       -0.0697565,  0.0000000,  0.9975641 ]
   
    data_array = data[['x','y','z']].to_numpy()
    data_rotated = np.dot(data_array, rot_mat.T)

    return data_rotated


def df_apply_rot(dataframe): 
    transformed_arr = rotate_pcl(dataframe)
    df_transformed = pd.DataFrame(transformed_arr, columns=['x', 'y', 'z'])
    df_transformed['intensity'] = dataframe['intensity']
    return df_transformed


def filter_ground_points(dataframe,  thresh):
    df_filtered = dataframe[dataframe.z <= thresh]
    return df_filtered

