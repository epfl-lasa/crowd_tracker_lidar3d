#!/usr/bin/env python
'''
    Pipeline for PointNet++ 
'''

import os 
import numpy as np 
import pandas as pd 

from crowd_tracker_lidar3d.hdf5_utils import load_h5, get_data_files

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/hdf5/")

class Dataset(): 
    def __init__(self, batch_size=10, normalize=False, intensity_channel=True, shuffle=None): 
        self.batch_size = batch_size
        self.normalize = normalize
        self.intensity_channel = intensity_channel 

        # load data files 
        path = DATA_DIR
        all_files = get_data_files(path)

        # TODO: shuffle frames 
        # TODO: create batches 

        data_batchlist, label_batchlist = [], []                                                                                       
        for f in all_files:                                                                                                            
            data, label = _load_data_file(os.path.join(BASE_DIR, f))                                                                   
            data_batchlist.append(data)                                                                                                
            label_batchlist.append(label)    
        data_batches = np.concatenate(data_batchlist, 0)
        label_batches = np.concatenate(label_batchlist,0)