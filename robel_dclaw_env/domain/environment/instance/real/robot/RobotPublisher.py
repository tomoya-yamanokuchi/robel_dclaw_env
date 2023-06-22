import rospy
import numpy as np
from std_msgs.msg import Int32, Int32MultiArray
from robel_dclaw_env.custom_service import angle_interface as ai


class RobotPublisher(object):
    def __init__(self):
        # Publisher initialization
        self.pub_initialize_ctrl = rospy.Publisher("/dclaw/initialize_ctrl/command",    Int32MultiArray,    queue_size=10)
        self.pub_joint_ctrl      = rospy.Publisher("/dclaw/joint_ctrl/command",         Int32MultiArray,    queue_size=10)
        self.pub_valve_ctrl      = rospy.Publisher("/dclaw/valve_ctrl/command",         Int32,              queue_size=10)
        # Message initialization
        self.msg_initialize_ctrl = Int32MultiArray()
        self.msg_joint_ctrl      = Int32MultiArray()
        self.msg_valve_ctrl      = Int32MultiArray()


    def publish_initialize_ctrl(self, ctrl):
        print(ctrl)
        joint_position  = ai.radian2resolution(ctrl[:9])
        position_p_gain = np.array(ctrl[9:], dtype=int)
        init_ctrl       = np.hstack([joint_position, position_p_gain])
        self.msg_initialize_ctrl.data = tuple(init_ctrl)
        print(self.msg_initialize_ctrl.data)
        self.pub_initialize_ctrl.publish(self.msg_initialize_ctrl)
        self.publish_joint_ctrl(ctrl[:9]) # <--必要：無いと初期化前に残っている制御入力に影響される


    def publish_joint_ctrl(self, ctrl):
        ctrl = ai.radian2resolution(ctrl)
        self.msg_joint_ctrl.data = tuple(ctrl)
        self.pub_joint_ctrl.publish(self.msg_joint_ctrl)


    def publish_valve_ctrl(self, ctrl):
        ctrl = ai.radian2resolution(ctrl)
        self.msg_valve_ctrl.data = tuple(ctrl)
        self.pub_valve_ctrl.publish(self.msg_valve_ctrl)
