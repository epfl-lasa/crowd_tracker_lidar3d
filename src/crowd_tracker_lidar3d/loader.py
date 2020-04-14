import os
import csv
import numpy as np

def load_data(filename, directory): 
    os.chdir(directory)

    input_file = filename + '.csv'

    # with open(input_file, 'r') as f:
    #     reader = csv.reader(f, delimiter=',')
    #     headers = next(reader)
    #     data = np.array(list(reader)).astype(float)

    data = np.genfromtxt(input_file, delimiter=',', names=True)
    
    print(headers)
    print(data.shape)

    return(data)