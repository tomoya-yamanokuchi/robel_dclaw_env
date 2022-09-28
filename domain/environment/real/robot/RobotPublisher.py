import rospy
import numpy as np
import domain.environment.real.robot.angle_interface as ai
from std_msgs.msg import Int32, Int32MultiArray



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
        joint_position  = ai.radian2resolution(ctrl[:9])
        position_p_gain = np.array(ctrl[9:], dtype=int)
        init_ctrl       = np.hstack([joint_position, position_p_gain])
        self.msg_initialize_ctrl.data = tuple(init_ctrl)
        self.pub_initialize_ctrl.publish(self.msg_initialize_ctrl)
        # self.publish_joint_ctrl(ctrl[:9])


    def publish_joint_ctrl(self, ctrl):
        ctrl = ai.radian2resolution(ctrl)
        self.msg_joint_ctrl.data = tuple(ctrl)
        self.pub_joint_ctrl.publish(self.msg_joint_ctrl)


    def publish_valve_ctrl(self, ctrl):
        ctrl = ai.radian2resolution(ctrl)
        self.msg_valve_ctrl.data = tuple(ctrl)
        self.pub_valve_ctrl.publish(self.msg_valve_ctrl)