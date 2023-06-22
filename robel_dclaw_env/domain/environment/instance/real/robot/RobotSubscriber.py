import numpy as np
import rospy
from std_msgs.msg import Int32, Int32MultiArray, Bool
from robel_dclaw_env.custom_service import angle_interface as ai



class RobotSubscriber(object):
    def __init__(self):
        # Subscriber initialization
        rospy.Subscriber("/dclaw/is_initialize_finished",   Bool,               self.callback_is_initialize_finished)
        rospy.Subscriber("/dclaw/joint_currents",           Int32MultiArray,    self.callback_joint_currents)
        rospy.Subscriber("/dclaw/joint_positions",          Int32MultiArray,    self.callback_joint_positions)
        rospy.Subscriber("/dclaw/joint_velocities",         Int32MultiArray,    self.callback_joint_velocities)
        rospy.Subscriber("/dclaw/valve_moving",             Int32,              self.callback_valve_moving)
        rospy.Subscriber("/dclaw/valve_position",           Int32,              self.callback_valve_position)
        # initialize subscribe variables
        self.is_initialize_finished = None
        self.joint_currents         = None
        self.joint_positions        = None
        self.joint_velocities       = None
        self.valve_moving           = None
        self.valve_position         = None
        # connection flag
        self.connection_flag = {}
        # self.connection_flag["is_initialize_finished"] = False
        self.connection_flag["joint_currents"]         = False
        self.connection_flag["joint_positions"]        = False
        self.connection_flag["joint_velocities"]       = False
        self.connection_flag["valve_moving"]           = False
        self.connection_flag["valve_position"]         = False
        self.subscribe_sum = len(self.connection_flag)


    def get_connection_flag_True_num(self):
        # print(np.sum(np.array(list(self.connection_flag.values()))*1))
        return np.sum(np.array(list(self.connection_flag.values()))*1)


    def callback_is_initialize_finished(self, data):
        self.is_initialize_finished = np.array(data.data)
        self.connection_flag["is_initialize_finished"] = True


    def callback_joint_currents(self, data):
        self.joint_currents = np.array(data.data)
        self.connection_flag["joint_currents"] = True


    def callback_joint_positions(self, data):
        data = ai.resolution2radian(np.array(data.data))
        self.joint_positions = data
        self.connection_flag["joint_positions"] = True


    def callback_joint_velocities(self, data):
        self.joint_velocities = np.array(data.data)
        self.connection_flag["joint_velocities"] = True


    def callback_valve_moving(self, data):
        self.valve_moving = np.array(data.data)
        self.connection_flag["valve_moving"] = True


    def callback_valve_position(self, data):
        data = ai.resolution2radian(np.array(data.data))
        self.valve_position = np.array(data.data)
        self.connection_flag["valve_position"] = True
