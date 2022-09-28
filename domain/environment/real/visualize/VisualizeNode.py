import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image


class VisualizeNode(object):
    def __init__(self):
        self.pub_observation = rospy.Publisher('/visualize/observation',  Image, queue_size=1)
        self.observation = None
        # --- connection flag ---
        self.connection_flag          = {}
        self.connection_flag["observation"] = False
        self.subscribe_sum            = len(self.connection_flag)


    def publish_observation(self, img):
        self.pub_observation.publish(ros_numpy.msgify(Image, img, encoding='bgr8'))


if __name__ == '__main__':
    node = VisualizeNode()
