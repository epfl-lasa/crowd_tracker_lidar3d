import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def rotate_pcl(data, quat=None): 
    """ Rotate point cloud. 

    Args:
        data (numpy ndarray): 3D point cloud. 
        quat (array): The quaternion to derive the rotation matrix from. Defaults to None.

    Returns:
        rotated point cloud 
    """
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
    """ Translate point cloud on the upward-facing axis. 

    Args:
        data (numpy ndarray): 3D point cloud. 
        z (float): translation value 

    Returns:
        translated data 
    """
    assert data.shape[0] >= data.shape[1] # assert that columns represent features in matrix/dataframe
    if isinstance(data, pd.DataFrame):
        data = data.add(z, axis='z')
    else: 
        data[:,3] + z
    return data

def df_apply_rot(dataframe, quat=None, return_full_df=False): 
    """ Apply rotation to given pandas dataframe of a point cloud. 

    Args:
        dataframe (pandas dataframe): point cloud 
        quat (array): The quaternion to derive the rotation matrix from. Defaults to None.
        return_full_df (bool, optional): If True, the dataframe will be return with all columns. 
                                         Otherwise  only columns {'x', 'y', 'z', 'intensity'} will be maintained. 
                                         Defaults to False.

    Returns:
        [type]: [description]
    """
    idx = dataframe.index
    transformed_arr = rotate_pcl(dataframe, quat)
    df_transformed = pd.DataFrame(transformed_arr, columns=['x', 'y', 'z'],  index=idx)
    if return_full_df: 
        dataframe['x'] = df_transformed['x']
        dataframe['y'] = df_transformed['y']
        dataframe['z'] = df_transformed['z']
        df_transformed = dataframe
    else: 
        df_transformed['intensity'] = dataframe['intensity']
    return df_transformed

def add_polar_coord(df): 
    """ Add columns representing polar coordinates 'r' and 'phi' to the point cloud dataframe. 

    Args:
        df (pandas dataframe): point cloud 

    Returns:
        dataframe_extended: extended dataframe by 'r' and 'phi'
    """
    dataframe = df.copy()
    dataframe['r'] = dataframe.apply(lambda row: np.hypot(row.x, row.y), axis=1) 
    dataframe['phi'] = dataframe.apply(lambda row: np.arctan2(row.y, row.x), axis=1)
    return dataframe

def return_ground_points(dataframe,  thresh):
    """ Filter point cloud by only ground plane. 

    Args:
        dataframe (pandas dataframe): point cloud 
        thresh (float): threshold on the z-axis, as of which to consider points as floor points.

    Returns:
        floor points
    """
    df_filtered = dataframe[dataframe.z <= thresh]
    return df_filtered

def remove_ground_points(dataframe,  thresh):
    """ Remove ground plane from point cloud. Assumption that ground is flat.  

    Args:
        dataframe (pandas dataframe): point cloud 
        thresh (float): threshold on the z-axis, as of which to consider points as floor points.

    Returns:
        filtered/reduced dataframe
    """
    df_filtered = dataframe[dataframe.z >= thresh].reset_index(drop=True)
    return df_filtered

def standardize_data(df):
    """ Standardize point cloud data. Dimensions x, y and z are standardized together, 
    whereas the intensity dimension is considered individually. 

    Args:
        df (pandas dataframe): point cloud 

    Returns:
        standardized df
    """
    normalized_df = df.copy()
    std_tot = np.sum([df.x.std(), df.y.std(), df.z.std()])
    for dim in ['x','y','z']:
        avg = df[dim].mean()
        normalized_df[dim] = (df[dim] - avg)/std_tot
    normalized_df['intensity'] =(df.intensity - df.intensity.mean()) / df.intensity.std()
    return normalized_df 


def preprocess_pipeline(df, radius_range, timestamp, quat, z_trans, floor_thresh):
    # Analyze pointcloud from timestamp chosen above 
    oneframe_data = df[df.rosbagTimestamp == timestamp].reset_index()
    # Rotate & translate points to compensate for LiDAR tilt angle 
    oneframe_data_trans = df_apply_rot(oneframe_data, quat, return_full_df=True)
    oneframe_data_trans = translate_height(oneframe_data_trans, z_trans)
    # Remove ground points
    oneframe_data_trans_no_floor = remove_ground_points(oneframe_data_trans, floor_thresh)
    # Filter by radius 
    oneframe_data_trans_no_floor = add_polar_coord(oneframe_data_trans_no_floor)
    label_mask = oneframe_data_trans_no_floor.r.between(*radius_range)
    oneframe_data_trans_no_floor['label'] = label_mask
    # Create template 
    template = oneframe_data_trans_no_floor[oneframe_data_trans_no_floor.label].reset_index(drop=True)
