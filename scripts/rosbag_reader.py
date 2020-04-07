#!/usr/bin/env python
'''
Extracts data from rosbags. 
Rosbags are stored in data folder within package, excluded from git repo.
'''

import csv		#writing CV files.
import yaml
import numpy as np
from rosbag.bag import Bag
import rosbag
import rospy
import os
import sys
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField

import sensor_msgs.point_cloud2 as pc2

import numpy as np

__author__ = "larabrudermueller"
__date__ = "2020-04-07"
__email__ = "lara.brudermuller@epfl.ch"

DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]

pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}



def fields_to_dtype(fields, point_step):
    '''
    Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

def pointcloud2_to_array(cloud_msg, squeeze=True):
        ''' Converts a rospy PointCloud2 message to a numpy recordarray
        Reshapes the returned array to have shape (height, width), even if the height is 1.
        The reason for using np.fromstring rather than struct.unpack is speed... especially
        for large point clouds, this will be <much> faster.
        '''
        # construct a numpy record type equivalent to the point type of this cloud
        dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

        # parse the cloud into an array
        cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

        if squeeze and cloud_msg.height == 1:
            return np.reshape(cloud_arr, (cloud_msg.width,))
        else:
            return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


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

        max_it_bag = 1
        it_bag = 0
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

    # def extract_lidar_data(self): 
    #     for pp in pc2.read_points(self.data_lidar, skip_nans=True, field_names=("x", "y", "z")):
    #         self.points_lidar_2d[:, it_count] = [pp[0], pp[1], pp[2]]

    

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
    
    print('shape msg', messages['/camera_front/depth/color/points'][0].shape)

    import pdb; pdb.set_trace() ### BREAKPOINT ###
    # pointcloud = messages['/front_lidar/velodyne_points']

    # Reader.extract_lidar_data()
    
    # with open("file.txt", "w") as output:
    #     output.write(str(pointcloud))
    # Reader.extract_data('/front_lidar/velodyne_points')
