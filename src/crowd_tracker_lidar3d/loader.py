import os
import csv
import numpy as np
import pandas as pd

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
