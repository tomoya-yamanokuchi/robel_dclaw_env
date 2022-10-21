import copy
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict

'''
This works fine on python 3.7 and higher versions.
'''

# gray = int((255 * 0.5) + 0.5)
# import ipdb; ipdb.set_trace()
floor_rgb               = np.array([200, 200, 200], dtype=np.uint8)
robot_rgb               = np.array([ 38,  38,  38], dtype=np.uint8)
finger_rgb              = np.array([255, 127,   0], dtype=np.uint8)
valve_3fin_handle_1_rgb = np.array([255, 0, 0], dtype=np.uint8)
valve_3fin_handle_2_rgb = np.array([0, 255, 0], dtype=np.uint8)
valve_3fin_handle_3_rgb = np.array([0, 0, 255], dtype=np.uint8)
valve_3fin_center_rgb   = np.array([255, 255, 255], dtype=np.uint8)


@dataclass(frozen=False)
class CanonicalRGB:
    rgb: Dict[str, int] = field(default_factory=lambda: (
        {
            "floor"                    : floor_rgb,
            # ------ robot ------
            # --> claw1
            "FFbase_xh28"               : robot_rgb,
            "FF10_metal_clamping"       : robot_rgb,
            "FF10_metal_clamping_small" : robot_rgb,
            "FF10_xh28"                 : robot_rgb,
            "FFL11_metal_clamping_small": robot_rgb,
            "FFL11_xh28"                : robot_rgb,
            "FFL11_metal_clamping"      : robot_rgb,
            "FFL12_metal_clamping"      : robot_rgb,
            "FFL12_plastic_finger"      : finger_rgb,
            # --> claw2
            "MFbase_xh28"               : robot_rgb,
            "MF20_metal_clamping"       : robot_rgb,
            "MF20_metal_clamping_small" : robot_rgb,
            "MF20_xh28"                 : robot_rgb,
            "MFL21_metal_clamping_small": robot_rgb,
            "MFL21_xh28"                : robot_rgb,
            "MFL21_metal_clamping"      : robot_rgb,
            "MFL22_metal_clamping"      : robot_rgb,
            "MFL22_plastic_finger"      : finger_rgb,
            # --> claw3
            "THbase_xh28"               : robot_rgb,
            "TH30_metal_clamping"       : robot_rgb,
            "TH30_metal_clamping_small" : robot_rgb,
            "TH30_xh28"                 : robot_rgb,
            "THL31_metal_clamping_small": robot_rgb,
            "THL31_xh28"                : robot_rgb,
            "THL31_metal_clamping"      : robot_rgb,
            "THL32_metal_clamping"      : robot_rgb,
            "THL32_plastic_finger"      : finger_rgb,
            # ------ valve ------
            "vis_valve_3fin_handle_1"   : valve_3fin_handle_1_rgb,
            "vis_valve_3fin_handle_2"   : valve_3fin_handle_2_rgb,
            "vis_valve_3fin_handle_3"   : valve_3fin_handle_3_rgb,
            "vis_valve_3fin_center"     : valve_3fin_center_rgb  ,
        }
    ))


if __name__ == '__main__':
    can = CanonicalRGB()
    print(can.rgb["THL31_metal_clamping"])
    print(can.rgb["FFL12_plastic_finger"])
    # a = can.asdict()
    # import ipdb; ipdb.set_trace()
