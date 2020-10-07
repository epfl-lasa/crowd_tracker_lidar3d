import numpy as np
import math


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calc_heading_angle(data, label_mask=None): 
    """ Calculate object rotation in Velodyne coordinates, i.e. the yaw angle, assmuing the other 
        two to be close to 0. Herefor, we calculate a PCA to translate XY-projection to most 
        dominant axis in 1D and then calculate angle from the x-axis to the principal component as 
        defined in the KITTI dataset paper.

    Args:
        data: numpy nd array containing either the entire point cloud data (loaded from hdf5 file) or the template itself
        label_mask: binary array of equal length as data to filter for detection points 

    Returns:
        orient_angle: the bounding box orientation in radians [-pi,pi]
        pca_stats: a dict with the pca results for plotting the principal component
    """
    scaler = StandardScaler()
    if label_mask is not None: 
        # template data for human 
        template = data[label_mask]
    else: 
        template=data
    # Standardize the features of human template 
    X_temp = StandardScaler().fit_transform(template[:,0:2])

    # Apply PCA 
    pca = PCA(n_components=1)
    pca.fit(X_temp)

    for v in pca.components_:       
        v_hat = v / np.linalg.norm(v) # make vector unit length
        end_pt = pca.mean_ + v_hat
        # Return the arc tangent of y/x in radians from -pi to pi 
        orient_angle = math.atan2(end_pt[1], end_pt[0]) # angle in radian   
        # print(math.degrees(orient_angle)) #angle in degrees 

    pca_stats = {
        'explained_variance': pca.explained_variance_ratio_, # percentage of variance explained by each of the selected components
        'eigenvalue': pca.explained_variance_, # largest eigenvalue of covariance matrix of data 
        'components': pca.components_, # principal axis in feature space, representing the directions of maximum variance in the data
        'mean': pca.mean_ # mean of the data used in PCA
    }

    return orient_angle, pca_stats

def boxes3d_to_corners3d_velodyne(boxes3d, rotate=True):
    """
    Calculate box corners given basic dimensions and centroid. 
    :param boxes3d: (N, 7) [x, y, z, h, w, l, angle]
    :param rotate: if true the box is rotated by the given angle ry around the up-axis (yaw)
    :return: corners3d: (N, 8, 3)
    """

    boxes_num = boxes3d.shape[0]
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([w / 2., w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2.], dtype=np.float32).T  # (N, 8)
    y_corners = np.array([-l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2.], dtype=np.float32).T  # (N, 8)
    z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dtype=np.float32).T  # (N, 8)

    if rotate:
        rz = boxes3d[:, 6]
        zeros, ones = np.zeros(rz.size, dtype=np.float32), np.ones(rz.size, dtype=np.float32)
        rot_list = np.array([[np.cos(rz), np.sin(rz), zeros],
                             [-np.sin(rz), np.cos(rz), zeros],
                             [zeros, zeros,  ones]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)