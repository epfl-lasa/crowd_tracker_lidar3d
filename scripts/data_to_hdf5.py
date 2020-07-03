#!/usr/bin/env python
'''
Saves data to one hdf5 file per frame/timestamp.

'''

import numpy as np
import os
import pandas as pd
import string

from crowd_tracker_lidar3d.loader import load_data_to_dataframe
from crowd_tracker_lidar3d.preprocessing import df_apply_rot, remove_ground_points, add_polar_coord
from crowd_tracker_lidar3d.hdf5_util import save_h5, load_h5

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/annotated/")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/hdf5/")
QUAT = [0, 0.0174524, 0, 0.9998477]
Z_TRANS = 0.480 


def get_data_filenames(): 
    data_files = []
    data_files = [str(f) for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.csv')]
    file_names = [string.rstrip(f, "_annotated.csv") for f in data_files]
    return sorted(file_names)

def main(): 
    print(DATA_DIR)
    file_names = get_data_filenames()
    print(file_names)
    for idx, f in enumerate(file_names):
        print('Processing file {}/{}: {}'.format(idx,len(file_names),f))
	    full_file = f + '_annotated'
    	data = load_data_to_dataframe(full_file, DATA_DIR)
        
    	# Rotate & Translate 
	        data_trans = df_apply_rot(data, QUAT, return_full_df=True)
        data_trans = data_trans.add(Z_TRANS, axis='z')
        
        # Save data per timeframe
        out_dir = SAVE_DIR + '/{}/'.format(f) #save frame files in separate directory
        if os.path.exists(out_dir):
            continue # File has already been processed
        os.makedirs(out_dir)
        for i,t in enumerate(data_trans.rosbagTimestamp.unique()): 
            data_temp = data_trans[data_trans.rosbagTimestamp==t].reset_index(drop=True)
            data_final = data_temp[['x', 'y', 'z', 'intensity']].to_numpy(dtype='float32')
            label = data_temp['label'].to_numpy(dtype='int')
            save_h5(out_dir + 'frame{}.h5'.format(i), data_final, label)

if __name__=='__main__':
    # print(os.listdir(DATA_DIR))
    main()
