#!/usr/bin/env python
'''
    Pipeline for PointNet++ 
'''

import os 
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset

from crowd_tracker_lidar3d.hdf5_utils import load_h5, get_data_files

# DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med/hdf5/")

class PointNetData(Dataset): 
    def __init__(self, root, batch_size=10, normalize=False, intensity_channel=True, shuffle=True): 
        """
        :param root: directory path to the dataset
        :param num_points: number of points to process for each scene
        :param normalize: whether include the normalized coords in features (default: False)
        :param intensity_channel: whether to include the intensity value to xyz coordinates (default: True)
        :param shuffek: whether to shuffle the data (default: True)
        """
        self.root = root
        self.batch_size = batch_size
        self.normalize = normalize
        self.intensity_channel = intensity_channel 
        self.shuffle = shuffle

        # load data files 
        all_files = get_data_files(self.root)

        # TODO: shuffle frames 
        # TODO: create batches 

        data_batchlist, label_batchlist = [], []                                                                                       
        for f in all_files:                                                                                                            
            data, labels = load_h5(f)                                                                   
            
            # TODO: check if needed here 
            # reshaping to size (n, 1) instead of (n,) because pytorch wants it like that
            labels_reshaped = np.ones((labels.shape[0], 1), dtype=np.float) # pylint: disable=E1136
            labels_reshaped[:, 0] = labels[:]
            labels = labels_reshaped
        
            data_batchlist.append(data)                                                                                                
            label_batchlist.append(labels)    
        data_batches = np.concatenate(data_batchlist, 0)
        label_batches = np.concatenate(label_batchlist,0)
        
        self.data = data_batchlist
        self.labels = label_batchlist

    def __len__(self):
        return len(self.data)

    def __getitem__(self):
        # TODO
        return None