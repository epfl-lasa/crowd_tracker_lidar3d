#!/usr/bin/env python
'''
Saves data to one hdf5 file per frame/timestamp.
'''

import numpy as np
import re 
import os
import pandas as pd
import string

from crowd_tracker_lidar3d.preprocessing import df_apply_rot, remove_ground_points, add_polar_coord
from crowd_tracker_lidar3d.hdf5_util import save_h5_basic, load_h5

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/LCAS/")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/LCAS/hdf5/")
CSV_DIR = '/hdd/master_lara_data/LCAS/'

# QUAT = [0, 0.0174524, 0, 0.9998477]
# Z_TRANS = 0.480 


def load_data_to_dataframe(filename, directory):
    """
    Loads data from csv file and returns it as pandas dataframe
    
    Args:
        filename (string)
        directory (string): path to directory containing correspodning csv file
    """
    input_file = os.path.join(directory, filename)
    data = pd.read_csv(input_file, header=None)
    # drop rows containing nans 
    data = data.dropna(axis=0, how='any').reset_index(drop=True)
    return(data)

def get_data_filenames(folder): 
    data_files = [str(f) for f in sorted(os.listdir(folder))]
    return sorted(data_files)

def main(): 
    label_dct = {}
    data_dct = {}
    reg_file = r'^([\w]+.[\w]+)'

    for _, dirs, files in os.walk(CSV_DIR):
        for folder in dirs: 
            time_span = folder.split('_')[2:4]
            folder_output_name = time_span[0] + '_' + time_span[1]
            out_dir = os.path.join(SAVE_DIR, folder_output_name) #save frame files in separate directory
            os.makedirs(out_dir,exist_ok=True)
            idx_map_file = os.path.join(SAVE_DIR, folder_output_name, 'idx_map.txt')
            data_files = get_data_filenames(os.path.join(CSV_DIR,folder))

            with open(idx_map_file, 'w') as out:
                for idx, f in enumerate(data_files): 
                    base_file_name = re.match(reg_file, f).group() 
                    out.write('{}:{}\n'.format(idx, base_file_name))
                    print('Processing file {}/{}: {}'.format(idx,len(data_files),base_file_name))
                    # load data
                    input_dir = os.path.join(CSV_DIR, folder)
                    data = load_data_to_dataframe(f, input_dir)
                    data_final = data.to_numpy(dtype='float32')
                    
                    save_h5_basic(os.path.join(out_dir, 'frame{}.h5'.format(idx)), data_final)


if __name__=='__main__':
    # print(os.listdir(DATA_DIR))
    main()