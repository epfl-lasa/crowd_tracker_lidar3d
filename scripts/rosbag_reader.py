#!/usr/bin/env python
'''
Extracts data from rosbags. 
Rosbags are stored in data folder within package, excluded from git repo.
'''

import csv		#writing CV files.
import yaml
import numpy as np
import rosbag
import rospy
import os
import sys
import string

import sensor_msgs.point_cloud2 as pc2
from rosbag.bag import Bag
from helpers import pointcloud2_to_array


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
    

    def save_bag_to_csv(self, topicList=None): 
        if not topicList: 
            topicList = self.readBagTopicList()

        for topic_name in topicList:
            bag_name = string.rstrip(self.bag.filename, ".bag")
            filename = '../data/'+ bag_name + string.replace(topic_name, '/', '-') + '.csv'
            
            # create csv file for topic data 
            with open(filename, 'w+') as csvfile:
                print("Parsing data. This may take a while.")
                filewriter = csv.writer(csvfile, delimiter = ',')

                for topic, msg, t in self.bag.read_messages(topic_name):	
                    # for each instance in time that has data for topicName
                    # parse data from this instance
                    print("current topic: {}".format(topic))
                    row = [str(t)]
                    # if type(msg).__name__ == '_sensor_msgs__PointCloud2':
                    if topic == '/front_lidar/velodyne_points':
                        header = ["rosbagTimestamp", "x", "y", "z", "intensity", "time"]
                        filewriter.writerow(header)
                        for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity", "time")):
                            row += [p[0], p[1], p[2], p[3], p[4]]
                            filewriter.writerow(row)
                    else: 
                        msg_fields = msg.fields
                        header = ['rosbagTimestamp']
                        for f in msg_fields: 
                            header.append(f.name)
                        filewriter.writerow(header)
                        for p in pc2.read_points(msg, skip_nans=True, field_names=tuple(header)):
                            row += list(p)
                            filewriter.writerow(row)

        self.bag.close()
        print("Parsed data. Saved as {}".format(filename))


    def read_bag(self, save_to_csv=True):
        """
        Return dict with all recorded messages with topics as keys
        Args:
            save_to_csv (bool, optional): Save data to csv files (one individual file per topic) if True
        Returns:
            dict: containing all published data points per topic
        """
        topics = self.readBagTopicList()
        messages = {}

        max_it_bag = 10
        it_bag = 0
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

            it_bag += 1
            if it_bag >= max_it_bag:
                break # stop -- too many topics

        self.bag.close()
        return messages

    def extract_lidar_data(self): 
        """
        extracts lidar data with help of helper functions
        """
        # TODO: probably not needed - erase 
        for topic, msg, _ in self.bag.read_messages(topics='/front_lidar/velodyne_points'):
            lidar_data = pointcloud2_to_array(msg)
        print(lidar_data)
        print(lidar_data.shape)

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
    bag_dir = os.path.dirname(os.path.abspath(__file__))
    bag_dir = os.path.join(bag_dir, "../data")
    input_bag = "1m_1person.bag"
    # input_bag = "3m_1person.bag"
    Reader = RosbagReader(bag_dir, input_bag)
    # topicList = Reader.readBagTopicList()
    # print topicList
    
    topics = ['/camera_front/depth/color/points', '/front_lidar/velodyne_points']
    Reader.save_bag_to_csv(topicList=topics)

    # messages = Reader.read_bag()
    # print('shape msg', messages['/camera_front/depth/color/points'][0].shape)
    # print('shape msg', messages['/front_lidar/velodyne_points'][0].shape)

    # import pdb; pdb.set_trace() ### BREAKPOINT ###
    # pointcloud = messages['/front_lidar/velodyne_points']

    # Reader.extract_lidar_data()
    
    # with open("file.txt", "w") as output:
    #     output.write(str(pointcloud))
