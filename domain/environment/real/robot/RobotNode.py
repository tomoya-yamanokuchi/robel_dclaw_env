import numpy as np
import rospy
from std_msgs.msg import Int32, Int32MultiArray, Bool

from .RobotSubscriber import RobotSubscriber
from .RobotPublisher import RobotPublisher


class RobotNode(object):
    def __init__(self):
        self.subscriber = RobotSubscriber()
        self.publisher  = RobotPublisher()
        self.wait_ros_connection_establishment()
        self.print_initialized_message()


    def wait_ros_connection_establishment(self):
        rospy.sleep(0.25)
        while self.subscriber.get_connection_flag_True_num() != self.subscriber.subscribe_sum:
            rospy.sleep(0.5)


    def print_initialized_message(self):
        rospy.loginfo("ROS connection is established! (Robot)")


if __name__ == '__main__':
    node = RobotNode()
