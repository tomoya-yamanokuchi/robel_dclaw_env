import copy
import numpy as np

'''
- 長さの単位: [mm]
'''

class KinematicsDefinition:
    def __init__(self):
        # ----- metal connection parts -----
        len_FR12_S102                  = 8.5  # 接地面からサーボ取り付け穴の中心の穴まで:./blueprint/FR12-S102.pdf
        len_FR12_H101                  = 28   # 接地面からサーボ取り付け穴の中心の穴まで:./blueprint/FR12-H101.pdf

        # ------- fingertip 3D parts -------
        # onshape URL: https://cad.onshape.com/documents/d707e0812c298911cddd6a01/w/0d3cd38cb2b52b0227025817/e/59153bf0b02469318b20a121
        radius_finger_tip_sphere       = 9.94410  # 指先の球体の部分の半径
        len_finger_tip_withdout_sphere = 45.72000 # 球体を除いた本体の長さ
        len_fingertip                  = radius_finger_tip_sphere + len_finger_tip_withdout_sphere

        # --------- dynamixel motor ---------
        # 繋パーツとのネジ穴からモータの中心軸位置まで:./blueprint/X430_dimension.pdf
        # (40 - 8 = 32)
        len_dynamixel                  = 32

        '''
            上記の数値を用いてアームの長さを計算
        '''
        self.l0 = len_FR12_H101 + len_FR12_S102 + len_dynamixel
        self.l1 = copy.deepcopy(self.l0)
        self.l2 = len_FR12_H101 + len_fingertip

        '''
            サーボ角度の限界（物理的にこれ以上はダメ，実測値ベース）
        '''

        self.margin = 0.005
        self.theta0_lb = -0.51        - self.margin
        self.theta0_ub = np.pi * 0.51 + self.margin

        self.theta1_lb = -1.7 - self.margin
        self.theta1_ub =  1.7 + self.margin

        self.theta2_lb = -2.0 - self.margin
        self.theta2_ub =  2.0 + self.margin



    def check_feasibility(self, theta):
        if len(theta.shape) == 1:
            theta = theta.reshape(1, -1)
        assert len(theta.shape) == 2
        assert theta.shape[-1] == 3

        theta0 = theta[:, 0]
        assert False not in (theta0 > self.theta0_lb), "\n\n <<<< [theta0_lb] input includes [{}] while lower_limit is [{}] >>>> \n\n".format(theta0.min(), self.theta0_lb)
        assert False not in (theta0 < self.theta0_ub), "\n\n <<<< [theta0_ub] input includes [{}] while upper_limit is [{}] >>>> \n\n".format(theta0.max(), self.theta0_ub)

        theta1 = theta[:, 1]
        assert False not in (theta1 > self.theta1_lb), "\n\n <<<< [theta1_lb] input includes [{}] while lower_limit is [{}] >>>> \n\n".format(theta1.min(), self.theta1_lb)
        assert False not in (theta1 < self.theta1_ub), "\n\n <<<< [theta1_ub] input includes [{}] while upper_limit is [{}] >>>> \n\n".format(theta1.max(), self.theta1_ub)

        theta2 = theta[:, 2]
        assert False not in (theta2 > self.theta2_lb), "\n\n <<<< [theta2_lb] input includes [{}], while lower_limit is [{}] >>>> \n\n".format(theta2.min(), self.theta2_lb)
        assert False not in (theta2 < self.theta2_ub), "\n\n <<<< [theta2_ub] input includes [{}], while upper_limit is [{}] >>>> \n\n".format(theta2.max(), self.theta2_ub)



    def show_info(self):
        print("--------------------------------")
        print("length:              ")
        print("       l0 = {:.3f}[mm]".format(self.l0))
        print("       l1 = {:.3f}[mm]".format(self.l1))
        print("       l2 = {:.3f}[mm]".format(self.l2))
        print("--------------------------------")
        print("joint limit:         ")
        print("   theta0 = [{: .2f}, {: .2f}][rad]".format(self.theta0_lb, self.theta0_ub))
        print("   theta1 = [{: .2f}, {: .2f}][rad]".format(self.theta1_lb, self.theta1_ub))
        print("   theta2 = [{: .2f}, {: .2f}][rad]".format(self.theta2_lb, self.theta2_ub))
        print("--------------------------------")


if __name__ == '__main__':
    kinematics = KinematicsDefinition()
    kinematics.show_info()
    kinematics.check_feasibility(np.array([kinematics.theta0_lb+0.01, 0.0, 0.0]))
