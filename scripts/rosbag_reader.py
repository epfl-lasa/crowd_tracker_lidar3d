#!/usr/bin/env python

'''
Extracts data from rosbags. 
Rosbags are stored in data folder within package, excluded from git repo.
'''

import csv		#writing CV files.
import yaml
import numpy as np
import rosbag
import cv2
# import pyrosbag
import rospy
import os
import sys
import string
import argparse
try: 
    import pcl 
except: 
    ModuleNotFoundError

import sensor_msgs.point_cloud2 as pc2
from rosbag.bag import Bag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

__author__ = "larabrudermueller"
__date__ = "2020-04-07"
__email__ = "lara.brudermuller@epfl.ch"


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--datadir', type=str, default='/hdd/data_qolo/med_crowd_06102020/', help='specify the dataset to use')
parser.add_argument('--input_bag', type=str, default=None, help='specify the rosbag to process')
args = parser.parse_args()

DATA_DIR = args.datadir


class RosbagReader(): 
    def __init__(self, bag_dir, input_bag, save_dir):
        self.save_dir =  save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.bag = Bag(os.path.join(bag_dir, input_bag))
    
    def print_bag_info(self):
        info_dict = yaml.load(Bag(input_bag, 'r')._get_yaml_info())
        print(info_dict)

    def extract_camera_data(self):
        image_topic = '/camera_left/color/image_raw'
        bag_name = self.bag.filename.strip(".bag").split('/')[-1]
        output_dir = os.path.join(self.save_dir,bag_name, 'camera')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        bridge = CvBridge()
        count = 0
        for topic, msg, t in self.bag.read_messages(topics=[image_topic]):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
            print("Wrote image {}".format(count))
            count += 1

    def save_bag_to_pcd(self, topicList=None): 
        if not topicList: 
            topicList = self.readBagTopicList()

        for topic_name in topicList:
            bag_name = self.bag.filename.strip(".bag").split('/')[-1]
            save_temp_dir = os.path.join(self.save_dir,bag_name)
            if not os.path.exists(save_temp_dir):
                os.makedirs(save_temp_dir)
            i = 0 
            for topic, msg, t in self.bag.read_messages(topic_name):	
                # for each instance in time that has data for topicName
                # parse data from this instance
                save_file = os.path.join(save_temp_dir,'frame_{}_{}.pcd'.format(i,t))
                i+=1

                if topic == '/front_lidar/velodyne_points':
                    pc_list = [[p[0], p[1], p[2]] for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))]
                print(np.array(pc_list, dtype=np.float32).shape)
                p = pcl.PointCloud(np.array(pc_list, dtype=np.float32))
                print(save_file)
                p.to_file(save_file)


    def save_bag_to_csv(self, topicList=None): 
        # os.chdir(self.save_dir)

        if not topicList: 
            topicList = self.readBagTopicList()

        for topic_name in topicList:
            bag_name = self.bag.filename.strip(".bag").split('/')[-1]
            filename = self.save_dir + bag_name + topic_name.replace('/', '-') + '.csv'
            
            # create csv file for topic data 
            with open(filename, 'w+') as csvfile:
                print("Parsing data. This may take a while.")
                filewriter = csv.writer(csvfile, delimiter = ',')
                flag = 1 # use flag to only write header once to csv file
                
                for topic, msg, t in self.bag.read_messages(topic_name):	
                    # for each instance in time that has data for topicName
                    # parse data from this instance

                    # print("current topic: {}".format(topic))
                    row = [t]
                    # if type(msg).__name__ == '_sensor_msgs__PointCloud2':
                    if topic == '/front_lidar/velodyne_points':
                        header = ["rosbagTimestamp", "x", "y", "z", "intensity", "time"]
                        if flag:
                            filewriter.writerow(header)
                            flag = 0
                        for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity", "time")):
                            row += [p[0], p[1], p[2], p[3], p[4]]
                            filewriter.writerow(row)
                            row = [t]

                    else: 
                        msg_fields = msg.fields
                        field_names = []
                        header = ['rosbagTimestamp']
                        for f in msg_fields: 
                            field_names.append(f.name)
                            header.append(f.name)
                        if flag: 
                            filewriter.writerow(header)
                            flag = 0
                        for p in pc2.read_points(msg, skip_nans=True, field_names=tuple(field_names)):
                            row += list(p)
                            filewriter.writerow(row)
                            row = [t]

        print("Parsed data. Saved as {}".format(filename))

    def save_rwth_detections(self): 
        """
        Saves Detections generated by RWTH Detector into separate csv files.
        """
        os.chdir(self.save_dir)
        topic_name='/drow_detected_persons'
        bag_name = string.rstrip(self.bag.filename, ".bag")
        filename = self.save_dir + bag_name + string.replace(topic_name, '/', '-') + '.csv'
        
        # create csv file for topic data 
        with open(filename, 'w+') as csvfile:
            print("Parsing data. This may take a while.")
            filewriter = csv.writer(csvfile, delimiter = ',')

            header = ['rosbagTimestamp', 'detection_id', 'confidence', 'det_x', 'det_y','det_z', 'height']
            filewriter.writerow(header)

            for _, msg, t in self.bag.read_messages(topic_name):
                for det in msg.detections:    
                    row = [t]
                    row.append(det.detection_id)
                    row.append(det.confidence)
                    row.append(det.pose.pose.position.x)
                    row.append(det.pose.pose.position.y)
                    row.append(det.pose.pose.position.z)
                    row.append(det.height)
                    filewriter.writerow(row)
                    row = [t]
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

        return messages

    def readBagTopicList(self):
        """
        Read and save the initial topic list from bag
        """
        print("[OK] Reading topics in this bag. Can take a while..")
        topicList = []
        bag_info = yaml.load(self.bag._get_yaml_info(), Loader=yaml.FullLoader)
        for info in bag_info['topics']:
            topicList.append(info['topic'])

        print('{0} topics found'.format(len(topicList)))
        return topicList


if __name__=='__main__':
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/med_new/")
    bag_dir = DATA_DIR

    if args.input_bag: 
        bag_files_list = [args.input_bag]
    else: 
        bag_files_list = [file_name for _,_,file_name in os.walk(bag_dir)].pop()

    for input_bag in bag_files_list: 
        print(input_bag)
        Reader = RosbagReader(bag_dir, input_bag, save_dir)
        topicList = Reader.readBagTopicList()
        print('-------- Topics -------- ')
        for topic in topicList: 
            print(topic)
        
        # topics = ['/camera_front/depth/color/points', '/front_lidar/velodyne_points']
        topics = ['/front_lidar/velodyne_points'] 
        # Reader.save_bag_to_csv(topicList=topics)
        # Reader.save_bag_to_pcd(topicList=topics)
        Reader.extract_camera_data()
        # Reader.save_rwth_detections()
    
        Reader.bag.close()

    # messages = Reader.read_bag()
    # print('shape msg', messages['/camera_front/depth/color/points'][0].shape)
    # print('shape msg', messages['/front_lidar/velodyne_points'][0].shape)

    # import pdb; pdb.set_trace() ### BREAKPOINT ###
    # pointcloud = messages['/front_lidar/velodyne_points']


