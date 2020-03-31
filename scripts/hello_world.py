#!/usr/bin/env python
'''
Example for usage of ROS with python
'''

import rospy
import rospkg
from std_msgs.msg import Bool

from multiprocessing import Lock

# import numpy as np

__author__ = "lukashuber"
__date__ = "2020-03-31"
__email__ = "lukas.huber@epfl.ch"

mutex = Lock() # Use lock to ensure that messages are not changed during aloop

class TestWritter():
    def __init__(self):
        rospy.init_node("MyNodeName")
        self.freq = 10 # [Hz]
        self.rate = rospy.Rate(self.freq)

        rospy.Subscriber("/topic_name", Bool, self.callback_function)

        self.pub_default = rospy.Publisher("/topic_name2", Bool, queue_size=10)

        self._data = None
        
        print("Init successful")
    
    def loop(self):
        while not rospy.is_shutdown():
            with mutex:

                msg_bool = Bool(0)
                self.pub_default.publish(msg_bool)
                print("Hello world. I just competed a loop.")
            
            self.rate.sleep() # Sleep outside of mutex to allow callback

    
    def callback_function(self, data):
        with mutex:
            self._data = data
            
            
if __name__=='__main__':
    try:
        test_writer = TestWritter()
        test_writer.loop()
        
    except rospy.ROSInterruptException:
        pass
