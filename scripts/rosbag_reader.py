'''
Extracts data from rosbags. 
Rosbags are stored in data folder within package, excluded from git repo.
'''

__author__ = "larabrudermueller"
__date__ = "2020-04-07"
__email__ = "lara.brudermuller@epfl.ch"

import argparse
import os

from pathlib import Path
from crowd_tracker_lidar3d.rosbag_reader import RosbagReader


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--datadir', type=str, default='/hdd/data_qolo/med_crowd_06102020/', help='specify the dataset to use')
parser.add_argument('--input_bag', type=str, default=None, help='specify the rosbag to process')
args = parser.parse_args()

DATA_DIR = Path(args.datadir)
SAVE_DIR = DATA_DIR / 'hdf5/'
print(SAVE_DIR.resolve())


if args.input_bag: 
    bag_list = [args.input_bag]
else: 
    bag_list = list(DATA_DIR.glob('*.bag'))  
    crowd_files = [f for f in bag_list if 'crowd' in str(f).split('/')[-1]]  

for input_bag in bag_list: 
    print(input_bag)  
    save_dir = SAVE_DIR / str(input_bag).split('/')[-1].split('.')[0]   
    Reader = RosbagReader(input_bag, save_dir)

#     # topicList = Reader.readBagTopicList()
#     # print('-------- Topics -------- ')
#     # for topic in topicList: 
#     #     print(topic)

    Reader.extract_lidar_frames('h5')

    Reader.bag.close()


