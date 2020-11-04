# !/usr/bin/env python
'''
Saves data to one hdf5 file per frame/timestamp.
'''

import numpy as np
import os
import pandas as pd
import string
import argparse

from crowd_tracker_lidar3d.loader import load_data_to_dataframe
from crowd_tracker_lidar3d.preprocessing import df_apply_rot, remove_ground_points, add_polar_coord
from crowd_tracker_lidar3d.hdf5_util import save_h5, save_h5_basic

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--datadir', type=str, default='../data/med/annotated/', help='specify the dataset to use')
parser.add_argument('--savedir', type=str, default='../data/med/hdf5/', help='specify where to save hd5 files')
parser.add_argument('--annotated', type=bool, default=False, help='Indicate if csv incudes label or not')
parser.add_argument('--sensor_mount', type=float, default=0.480, help='Indicate the height at which the LiDAR sensor was mounted')


args = parser.parse_args()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.datadir)
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.savedir)
QUAT = [0, 0.0174524, 0, 0.9998477]
Z_TRANS = args.sensor_mount 


def get_data_filenames(raw=True): 
    data_files = []
    data_files = [str(f) for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.csv')]
    if not raw: 
        file_names = [f.strip("_annotated.csv") for f in data_files]
    else: 
        file_names = [f.split('.')[0] for f in data_files]
    return sorted(file_names)

def main(): 
    print(DATA_DIR)
    file_names = get_data_filenames()
    print(file_names)
    for idx, f in enumerate(file_names):
        print('Processing file {}/{}: {}'.format(idx,len(file_names),f))
        if args.annotated: f += '_annotated'
        data = load_data_to_dataframe(f, DATA_DIR)
        
    	# Rotate & Translate 
        data_trans = df_apply_rot(data, QUAT, return_full_df=True)
        data_trans['z'] += Z_TRANS 
        
        # Save data per timeframe
        out_dir = os.path.join(SAVE_DIR, f) #save frame files in separate directory
        # if os.path.exists(out_dir):
        #     continue # File has already been processed
        os.makedirs(out_dir, exist_ok=True)
        for i,t in enumerate(data_trans.rosbagTimestamp.unique()): 
            file_name = os.path.join(out_dir, 'frame{}.h5'.format(i))
            data_temp = data_trans[data_trans.rosbagTimestamp==t].reset_index(drop=True)
            data_final = data_temp[['x', 'y', 'z', 'intensity']].to_numpy(dtype='float32')
            if args.annotated: 
                label = data_temp['label'].to_numpy(dtype='int')
                save_h5(file_name, data_final, label)
            else: 
                save_h5_basic(file_name, data_final)

if __name__=='__main__':

    print(os.listdir(DATA_DIR))
    main()