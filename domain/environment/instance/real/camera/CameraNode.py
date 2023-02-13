import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image


class CameraNode(object):
    def __init__(self):
        rospy.Subscriber('/camera/image_raw', Image, self.callback_img)
        self.image = None
        # --- connection flag ---
        self.connection_flag          = {}
        self.connection_flag["image"] = False
        self.subscribe_sum            = len(self.connection_flag)
        self.wait_ros_connection_establishment()
        self.print_initialized_message()


    def wait_ros_connection_establishment(self):
        rospy.sleep(0.25)
        while self.get_connection_flag_True_num() != self.subscribe_sum:
            rospy.sleep(0.5)


    def print_initialized_message(self):
        rospy.loginfo("ROS connection is established! (Camera)")


    def get_connection_flag_True_num(self):
        print( np.sum(np.array(list(self.connection_flag.values()))*1))
        return np.sum(np.array(list(self.connection_flag.values()))*1)


    def callback_img(self, img):
        self.image = ros_numpy.numpify(img)
        self.connection_flag["image"] = True


if __name__ == '__main__':
    node = CameraNode()
