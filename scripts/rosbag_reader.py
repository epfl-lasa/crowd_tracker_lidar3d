#!/usr/bin/env python
'''
Extracts data from rosbags. 
Rosbags are stored in data folder within package, excluded from git repo.
'''

import csv		#writing CV files.
import yaml
from rosbag.bag import Bag
import rosbag
import rospy
import os
import sys

import sensor_msgs.point_cloud2 as pc2

import numpy as np

__author__ = "larabrudermueller"
__date__ = "2020-04-07"
__email__ = "lara.brudermuller@epfl.ch"

class RosbagReader(): 
    def __init__(self, bag_dir, input_bag):
        os.chdir(bag_dir)
        self.bag = rosbag.Bag(input_bag)
    
    #TODO: change to arguments instead of hardcoding directory 
    # inputFileName = sys.argv[1]
    # print "[OK] Found bag: %s" % inputFileName

    def print_bag_info(self):
        info_dict = yaml.load(Bag(input_bag, 'r')._get_yaml_info())
        print(info_dict)
    
    def read_bag(self):
        """
        Return dict with all recorded messages with topics as keys
        """
        topics = self.readBagTopicList()
        messages = {}
        # iterate through topic, message, timestamp of published messages in bag
        
        for topic, msg, _ in self.bag.read_messages(topics=topics):
            
            if type(msg).__name__ == '_sensor_msgs__PointCloud2':
                points = np.zeros((msg.height*msg.width, 3))
                for pp, ii in zip(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")),
                                  range(points.shape[0])):
                    points[ii, :] = [pp[0], pp[1], pp[2]]
                msg = points
            
            if topic not in messages:
                messages[topic] = [msg]
            else:
                messages[topic].append(msg)
        self.bag.close()
        return messages

    def readBagTopicList(self):
        """
        Read and save the initial topic list from bag
        """
        print "[OK] Reading topics in this bag. Can take a while.."
        topicList = []
        bag_info = yaml.load(self.bag._get_yaml_info())
        for info in bag_info['topics']:
            topicList.append(info['topic'])

        print '{0} topics found'.format(len(topicList))
        return topicList

if __name__=='__main__':
    print(os.getcwd())
    bag_dir = os.path.dirname(os.path.abspath(__file__))
    bag_dir = os.path.join(bag_dir, "../data")
    # bag_dir = "data"
    input_bag = "1m_1person.bag"
    # input_bag = "3m_1person.bag"
    Reader = RosbagReader(bag_dir, input_bag)
    # topicList = Reader.readBagTopicList()
    # print topicList
    messages = Reader.read_bag()
    pointcloud = messages['/front_lidar/velodyne_points']

    # with open("file.txt", "w") as output:
    #     output.write(str(pointcloud))
    # Reader.extract_data('/front_lidar/velodyne_points')
