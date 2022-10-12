import numpy
import rospy
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.real.robot.RobotNode import RobotNode
from custom_service import angle_interface as ai

ctrl_init_positions = np.array(
    [
        0.0, -0.0, -0.0,
        0.0, -0.0, -0.0,
        0.0, -0.0, -0.0,
    ]
)

resol = ai.radian2resolution(ctrl_init_positions)
# import ipdb; ipdb.set_trace()

rospy.init_node("ros_debug", anonymous=True)
robot_node           = RobotNode()
claw_Position_P_Gain = np.array([30, 30, 30], dtype=int)
init_command         = np.hstack([ctrl_init_positions, claw_Position_P_Gain])
robot_node.publisher.publish_initialize_ctrl(init_command)

rospy.sleep(1)

import ipdb; ipdb.set_trace()

while not robot_node.subscriber.is_initialize_finished:
    time.sleep(0.1)
time.sleep(2)


ctrl_radian = np.array(
    [
        0.02, -0.2, -0.2,
        0.02, -0.2, -0.2,
        0.02, -0.2, -0.2,
    ]
)
robot_node.publisher.publish_joint_ctrl(ctrl_radian)


rospy.sleep(1)