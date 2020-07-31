import numpy as np
import math


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calc_heading_angle(data, label_mask): 
    """ Calculate object rotation in Velodyne coordinates, i.e. the yaw angle, assmuing the other 
        two to be close to 0. Herefor, we calculate a PCA to translate XY-projection to most 
        dominant axis in 1D and then calculate angle from the x-axis to the principal component as 
        defined in the KITTI dataset paper.

    Args:
        data: numpy nd array containing point cloud data (loaded from hdf5 file)
        label_mask: binary array of equal length as data to filter for detection points 

    Returns:
        orient_angle: the bounding box orientation in radians [-pi,pi]
        pca_stats: a dict with the pca results for plotting the principal component
    """
    scaler = StandardScaler()
    # template data for human 
    template = data[label_mask]
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