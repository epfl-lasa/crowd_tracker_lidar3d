#!/usr/bin/env python

'''

Annotates data from csv files with human template models (when not walking).

'''
import numpy as np
import os
import pandas as pd
import string

from crowd_tracker_lidar3d.loader import load_data_to_dataframe
from crowd_tracker_lidar3d.preprocessing import df_apply_rot, remove_ground_points, add_polar_coord

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/")
QUAT = [0, 0.0174524, 0, 0.9998477]
Z_TRANS = 0.480 


def get_data_filenames(): 
    data_files = []
    data_files = [str(f) for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.csv')]
    file_names = [string.rstrip(f, "-front_lidar-velodyne_points.csv") for f in data_files]
    return file_names

def main(): 
    print(DATA_DIR)
    file_names = get_data_filenames()
    static_files = [f for f in sorted(file_names) if f.startswith('person')]
    print(static_files)
    i = 0
    for f in static_files: 
        if i == 1: 
            break
        print("Processing file: {}\n".format(f))
        full_file = f + '-front_lidar-velodyne_points'
        data = load_data_to_dataframe(full_file, DATA_DIR)
        dist = float(f.rsplit('_')[1])
        interval = [dist-0.5, dist+1]
        
        # Rotate & Translate 
        data_trans = df_apply_rot(data, QUAT, return_full_df=True)
        data_trans = data_trans.add(Z_TRANS, axis='z')

        # Add polar coordinates & filter for label 
        data_trans = add_polar_coord(data_trans)
        label_mask = (data_trans.r.between(*interval) & data_trans.y.between(-0.3,1))
        data['label'] = label_mask
        
        # Save annotated data 
        data.to_csv(DATA_DIR + '/annotated/' + f + '_annotated.csv', index = False)
        i +=1

if __name__=='__main__':
    # print(os.listdir(DATA_DIR))
    main()
