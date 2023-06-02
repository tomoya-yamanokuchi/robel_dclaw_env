import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from kinematics.domain import NumpyInverseKinematics, TorchInverseKinematics
from kinematics.service import to_numpy, to_tensor
import numpy as np


class Test:
    def __init__(self):
        self.numpy_inverse = NumpyInverseKinematics()
        self.torch_inverse = TorchInverseKinematics()


    def test_inverse_kenimatics_1claw(self, joint_position):
        out_numpy = self.numpy_inverse.calc_1claw(joint_position)
        out_torch = self.torch_inverse.calc_1claw(to_tensor(joint_position).cuda())
        # import ipdb; ipdb.set_trace()
        np.testing.assert_almost_equal(
            actual  = out_numpy,
            desired = to_numpy(out_torch),
            decimal = 4
        )
        print("<< pass unit test 1claw >>")


    def test_inverse_kenimatics_all_claw(self, joint_position):
        out_numpy = self.numpy_inverse.calc(joint_position)
        out_torch = self.torch_inverse.calc(to_tensor(joint_position).cuda())
        np.testing.assert_almost_equal(
            actual  = out_numpy,
            desired = to_numpy(out_torch),
            decimal = 4
        )
        print("<< pass unit test all claw>>")


if __name__ == '__main__':
    test = Test()


    endeffector_position = np.array(
        [
            [220.664, 0.0, 0.0],
            # [68.5, -151.0, 0.0],
            # [152.1641, -68.5, 0.],
            # [152.1641, -100, 0.],
        ]
    )
    test.test_inverse_kenimatics_1claw(endeffector_position)


    endeffector_position = np.array(
        [
            [220.664, 0.0, 0.0, 220.664, 0.0, 0.0, 220.664, 0.0, 0.0],
        ]
    )
    test.test_inverse_kenimatics_all_claw(endeffector_position)


