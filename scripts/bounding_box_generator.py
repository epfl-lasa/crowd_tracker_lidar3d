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
import shutil

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from crowd_tracker_lidar3d.loader import load_data_to_dataframe
from crowd_tracker_lidar3d.preprocessing import df_apply_rot, remove_ground_points, add_polar_coord
from crowd_tracker_lidar3d.hdf5_util import save_h5, load_h5
from crowd_tracker_lidar3d.annotation_utils import calc_heading_angle

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/hdf5/")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/annotated_boundbox/")


    
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
            shutil.rmtree(SAVE_DIR)
            # continue # File has already been processed
        os.makedirs(out_dir)
        print("Processing {} frames".format(len(data_files)))
        
        # iterate over labeled point cloud frames saved in hdf5 files 
        empy_frames = 0 
        for idx, f in enumerate(data_files): 
            # print('Processing file {}/{}: {}'.format(idx,len(data_files),f))
            full_file = os.path.join(path, f)
            data, label = load_h5(full_file) 

            final_data = data
            print('Number of points: {}'.format(data.shape[0]))
            # label = label[ground_mask]

            # filter data for only positive labels and do bbox calculations only on mask
            mask = np.where(label) 
            filter_data = final_data[mask]

            # remove floor points from data 
            ground_mask = np.where(filter_data[:,2] >= 0)
            filter_data = filter_data[ground_mask]

            # Check if frame contains human/annotations
            if filter_data.shape[0] == 0: 
                empy_frames += 1
                continue 

            # calculate centroid from pointcloud only using spatial coordinates
            centroid = filter_data[:,:3].mean(axis=0)

            min_x, min_y, min_z = np.min(filter_data[:,0]), np.min(filter_data[:,1]), np.min(filter_data[:,2])
            max_x, max_y, max_z = np.max(filter_data[:,0]), np.max(filter_data[:,1]), np.max(filter_data[:,2])

            # create bounding box parameters (h,w,l)
            h = max_z - np.max(min_z, 0) # assume that legs always on ground  
            w = max_x - min_x
            l = max_y - min_y

            # enlarge bbox by a small constant 
            const = 0.1
            h = h + const 
            w = w + const 
            l = l + const 

            # Compute orientation 
            orient_angle, pca = calc_heading_angle(final_data, label)
            # pca_stats = [pca['mean'], pca['components']]

            bbox = np.concatenate((centroid, (h,w,l), ([orient_angle])))
            save_h5(os.path.join(out_dir,f), final_data, label, bbox=bbox)

        print("{}/{} frames empty.".format(empy_frames, len(data_files)))

if __name__=='__main__':
    main()