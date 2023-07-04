import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from robel_dclaw_env.custom_service import angle_interface as ai

'''
- 3次元空間で1次元の拘束されたtask_space構築するのに必要となる基準点
'''

class ReferencePosition:
    def __init__(self):
        self.reference_resoluation = [
            [1860, 2048, 2048],
            [2048, 2048, 2048],
            [2238, 2300, 2048],
            [2312, 2048, 2048],
            [2224, 1766, 2048],
            [1967, 1766, 2048],
        ]
        self.num_reference = len(self.reference_resoluation)

        # 各基準点の間を補完する点の数
        # 6個基準点があるのでそれに合わせる
        # self.num_augument = [1, 1, 1, 1, 1, 1]
        # self.num_augument = [5, 5, 4, 4, 3, 3]
        self.num_augument = [50, 80, 50, 50, 50, 70]


    def augument_reference_resoluation(self):
        traj_idx = [[0] * self.num_reference for i in range(3)]
        for idx in range(len(traj_idx)):
            for i in range(self.num_reference):
                current_parent_basis_joint_position = self.reference_resoluation[i][idx]
                next_parent_basis_joint_position    = np.take(self.reference_resoluation, i+1, axis=0, mode='wrap')[idx]
                traj_idx[idx][i] = np.linspace(
                        start    = current_parent_basis_joint_position,
                        stop     = next_parent_basis_joint_position,
                        num      = self.num_augument[i],
                        endpoint = False
                )
            traj_idx[idx] = np.concatenate(traj_idx[idx])
        augumented_reference_resoluation = np.stack(traj_idx, axis=-1)
        return augumented_reference_resoluation


    def augument_reference_joint_position(self):
        return ai.resolution2radian(self.augument_reference_resoluation())


if __name__ == '__main__':
    c = ReferencePosition()
    augumented_reference_resoluation = c.augument_reference_joint_position()
    print(augumented_reference_resoluation)
