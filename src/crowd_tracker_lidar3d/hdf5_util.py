import os 
import sys 
import numpy as np 
import scipy.io as sio
import h5py

'''
    Helper functions to write .h5 data files for pointnet, etc.
'''

def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='int'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype,
    )
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype,
    )
    h5_fout.close()
    

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    # f.keys() should be [u'data', u'label']
    data = f['data'][:]
    label = f['label'][:]
    return (data,label)


def get_data_files(path): 
    data_files = []                                    
    for root, dirs, files in os.walk(path):
         for file in files:
             if file.endswith('.h5'):
                data_files.append(file) 
    return data_fi