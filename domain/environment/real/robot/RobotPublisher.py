import rospy
import numpy as np
from std_msgs.msg import Int32, Int32MultiArray
from custom_service import angle_interface as ai


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


    def __print_initialization_info(self, msg_initialize_ctrl):
        print("\n\n")
        print("=====================================================================================")
        print("  initial joint_positions : ", msg_initialize_ctrl.data[:9])
        print("            Current_Limit : ", msg_initialize_ctrl.data[9:18])
        print("          Position_P_Gain : ", msg_initialize_ctrl.data[18:])
        print("=====================================================================================")
        print("\n\n")


    def publish_initialize_ctrl(self, ctrl, current_limit, position_P_Gain):
        assert            ctrl.shape == (9,), print("ctrl.shape[0] == ", ctrl.shape[0])
        assert   current_limit.shape == (9,), print("current_limit.shape[0] == ", current_limit.shape[0])
        assert position_P_Gain.shape == (9,), print("position_P_Gain.shape[0] == ", position_P_Gain.shape[0])
        joint_position = ai.radian2resolution(ctrl)
        init_ctrl      = np.hstack([joint_position, current_limit, position_P_Gain])
        self.msg_initialize_ctrl.data = tuple(init_ctrl)
        self.__print_initialization_info(self.msg_initialize_ctrl)
        self.pub_initialize_ctrl.publish(self.msg_initialize_ctrl)
        self.publish_joint_ctrl(ctrl[:9]) # <--必要：無いと初期化前に残っている制御入力が入力され続けてしまう


    def publish_joint_ctrl(self, ctrl):
        ctrl = ai.radian2resolution(ctrl)
        self.msg_joint_ctrl.data = tuple(ctrl)
        self.pub_joint_ctrl.publish(self.msg_joint_ctrl)


    def publish_valve_ctrl(self, ctrl):
        ctrl = ai.radian2resolution(ctrl)
        self.msg_valve_ctrl.data = tuple(ctrl)
        self.pub_valve_ctrl.publish(self.msg_valve_ctrl)