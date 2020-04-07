#!/usr/bin/env python
'''
Extracts data from rosbags
'''

import yaml
from rosbag.bag import Bag
import rosbag
import os
import sys

__author__ = "larabrudermueller"
__date__ = "2020-04-07"
__email__ = "lara.brudermuller@epfl.ch"

class RosbagReader(): 
    def __init__(self, bag_dir, input_bag):
        os.chdir(bag_dir)
        self.input_bag = input_bag

    def print_bag_info(self):

        info_dict = yaml.load(Bag(self.input_bag, 'r')._get_yaml_info())
        print(info_dict)

    def read_bag(self):
        with open(os.path.join(sys.path[0],'./topic_list.txt')) as f: 
            topics  = [line for line in f.read().split(',\n')]
        print(topics)

        bag = rosbag.Bag(self.input_bag)
        messages = {}
        # iterate through topic, message, timestamp of published messages in bag 
        for topic, msg, _ in bag.read_messages(topics=topics):
            if topic not in messages:
                messages[topic] = [msg]
            else:
                messages[topic].append(msg)
        bag.close()

    def readBagTopicList(self):
        """
        Read and save the initial topic list from bag
        """
        print "[OK] Reading topics in this bag. Can take a while.."
        topicList = []
        bag = rosbag.Bag(self.input_bag)
        bag_info = yaml.load(bag._get_yaml_info())
        for info in bag_info['topics']:
            topicList.append(info['topic'])

        print '{0} topics found'.format(len(topicList))
        return topicList


if __name__=='__main__':
    bag_dir = "/home/brudermueller/Documents/lidar_tracking"
    input_bag = "3m_1person.bag"
    Reader = RosbagReader(bag_dir, input_bag)
    topicList = Reader.readBagTopicList()
    print topicList