import os
import csv
import numpy as np
import pandas as pd
import open3d as o3d 


def load_data(filename, directory): 
    """
    Loads and returns data in form of a numpy nd array from a csv file. 
    The msg fields should be the csv headers. The data can the be accessed 
    by calling "data['msgField']".
    
    Args:
        filename (string)
        directory (string): path to directory containing correspodning csv file
    """
    os.chdir(directory)
    input_file = filename + '.csv'

    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)

    data = np.genfromtxt(input_file, delimiter=',', names=True)
    print(headers)
    print(data.shape)
    return(data)

def load_data_to_dataframe(filename, directory):
    """
    Loads data from csv file and returns it as pandas dataframe
    
    Args:
        filename (string)
        directory (string): path to directory containing correspodning csv file
    """
    os.chdir(directory)
    input_file = filename + '.csv'
    data = pd.read_csv(input_file)
    # drop rows containing nans 
    data = data.dropna(axis=0, how='any')
    return(data)

def load_pcd_file(filename, path=None): 
    """
    Loads pcd files to numpy array with xyz values. 
    """
    if path == None: # take LCAS pcd files as default  
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/LCAS/LCAS_20160523_1200_1218_pcd/")
        pc = os.path.join(data_path, filename)
    else: # filename includes path 
        pc = filename 

    pcd = o3d.io.read_point_cloud(pc)
    pcd_array = np.asarray(pcd.points)
    print(np.asarray(pcd.colors))
    return(pcd_array)
