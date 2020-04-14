#!/usr/bin/env python
'''
Extracts data from rosbags. 
Rosbags are stored in data folder within package, excluded from git repo.
'''
import os
import sensor_msgs.point_cloud2 as pc2

from crowd_tracker_lidar3d.loader import load_data

if __name__=='__main__':
    bag_dir = os.path.dirname(os.path.abspath(__file__))
    bag_dir = os.path.join(bag_dir, "../data")
    data = load_data('1m_1person-front_lidar-velodyne_points', bag_dir)
    import pdb; pdb.set_trace() ### BREAKPOINT ###
