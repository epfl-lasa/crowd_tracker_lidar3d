#!/usr/bin/env python
'''
    Adds approximate minimum 3D bounding box to data labels. Encoded with 6 DOF, 
    i.e. position of box centroid (x,y,z) as well as box dimensions (h,w,l) and 
    heading angle (ry) measured around y axis.
'''

import numpy as np
import math
import os
import pandas as pd
import string

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from crowd_tracker_lidar3d.loader import load_data_to_dataframe
from crowd_tracker_lidar3d.preprocessing import df_apply_rot, remove_ground_points, add_polar_coord
from crowd_tracker_lidar3d.hdf5_util import save_h5, load_h5

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/hdf5/")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/annotated_boundbox/")


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
    # Standardizing the features
    X = StandardScaler().fit_transform(data[:,0:2])
    # tenplate data for human 
    X_temp = X[label_mask]

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
        'components': pca.components_, # principal axes in feature space, representing the directions of maximum variance in the data
        'mean': pca.mean_ # mean of the data used in PCA
    }

    return orient_angle, pca_stats

    
def main(): 
    print(DATA_DIR)
    # iterate over folders, each containing a data recording scene
    for folder in sorted(os.listdir(DATA_DIR)):
        print("------------------ Sequence: {} ------------------\n".format(folder))
        path = os.path.join(DATA_DIR, folder) 
        data_files = [str(f) for f in sorted(os.listdir(path)) if f.endswith('.h5')] 
        
        # Save data per timeframe
        out_dir = os.path.join(SAVE_DIR, folder) #save files in separate directory
        if os.path.exists(out_dir):
            continue # File has already been processed
        os.makedirs(out_dir)
        print("Processing {} frames".format(len(data_files)))
        
        # iterate over labeled point cloud frames saved in hdf5 files 
        empy_frames = 0 
        for idx, f in enumerate(data_files): 
            # print('Processing file {}/{}: {}'.format(idx,len(data_files),f))
            full_file = os.path.join(path, f)
            data, label = load_h5(full_file) 

            # remove floor points from data 
            ground_mask = np.where(data[:,2] >= 0)
            final_data = data[ground_mask]
            label = label[ground_mask]

            # filter data for only positive labels and do bbox calculations only on mask
            mask = np.where(label) 
            filter_data = final_data[mask]

            # Check if frame contains human/annotations
            if filter_data.shape[0] == 0: 
                empy_frames += 1
                continue 

            # calculate centroid from pointcloud only using spatial coordinates
            centroid = filter_data[:,:3].mean(axis=0)

            min_x, min_y, min_z = np.min(filter_data[:,0]), np.min(filter_data[:,1]), np.min(filter_data[:,2])
            max_x, max_y, max_z = np.max(filter_data[:,0]), np.max(filter_data[:,1]), np.max(filter_data[:,2])

            # create bounding box parameters (h,w,l)
            h = max_z - 0 # assume that legs always on ground  
            w = max_x - min_x
            l = max_y - min_y

            # enlarge bbox by a small constant 
            const = 0.1
            h = h + const 
            w = w + const 
            l = l + const 

            # Compute orientation 
            orient_angle, pca = calc_heading_angle(final_data, label)
            pca_stats = [pca['mean'], pca['components']]
            bbox = np.concatenate((centroid, (h,w,l), ([orient_angle])))
            save_h5(os.path.join(out_dir,f), final_data, label, bbox=bbox)

        print("{}/{} frames empty.".format(empy_frames, len(data_files)))

if __name__=='__main__':
    main()