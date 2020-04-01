import yaml
from rosbag.bag import Bag
import rosbag
import os
import sys

def print_bag_info(input_bag):

    info_dict = yaml.load(Bag(input_bag, 'r')._get_yaml_info())
    print(info_dict)

def read_bag(input_bag):
    with open(os.path.join(sys.path[0],'./topic_list.txt')) as f: 
        topics  = [line for line in f.read().split(',\n')]
    print(topics)

    bag = rosbag.Bag(input_bag)
    messages = {}
    # iterate through topic, message, timestamp of published messages in bag 
    for topic, msg, _ in bag.read_messages(topics=topics):
        if topic not in messages:
            messages[topic] = [msg]
        else:
            messages[topic].append(msg)
    bag.close()

def readBagTopicList(bag):
    """
    Read and save the initial topic list from bag
    """
    print "[OK] Reading topics in this bag. Can take a while.."
    topicList = []
    bag = rosbag.Bag(input_bag)
    bag_info = yaml.load(bag._get_yaml_info())
    for info in bag_info['topics']:
        topicList.append(info['topic'])

    print '{0} topics found'.format(len(topicList))
    return topicList


bag_dir = "/home/brudermueller/Documents/lidar_tracking"
os.chdir(bag_dir) 
input_bag = "3m_1person.bag"
# read_bag(input_bag)
topicList = readBagTopicList(input_bag)
print topicList