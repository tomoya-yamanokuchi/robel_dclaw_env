import numpy as np


class CtrlInterpolation:
    '''
    ・kpのゲインを20とかの大きめの値にしてると inplicit step を回す時に
    目標角度がそこまで離れていなくても指が急速に動いてobjectを弾いてしまう

    ・inplicit step を回す時に目標値（制御入力の値）をなめからに補完することで
    大きめのpositionゲインを保ったまま指がゆっくりと動くことを保証する

    ・最終的には 目標値に到達していてほしいので endpoint_margin_step で設定した
    ステップ数文はもとの目標値になるように余裕をもたせる
    '''

    def __init__(self, num_interpolation, endpoint_margin_step):
        self.dim_ctrl             = 9
        self.num_interpolation    = num_interpolation
        self.endpoint_margin_step = endpoint_margin_step


    def concatenate(self, w_now, w_add):
        if w_now is None: return w_add
        return np.concatenate((w_now, w_add), axis=1)


    def interpolate(self, current_joint_position, target_joint_position):
        assert current_joint_position.shape == (self.dim_ctrl,)
        assert target_joint_position.shape  == (self.dim_ctrl,)

        interpolated_ctrl = None
        for d in range(self.dim_ctrl):
            ctrl_d_until_endpoint  = self.__interpolate_until_endpoint(current_joint_position[d], target_joint_position[d])
            ctrl_d_endpoint_margin = self.__interpolate_endpoint_margin(target_joint_position[d])
            ctrl_d                 = np.concatenate((ctrl_d_until_endpoint, ctrl_d_endpoint_margin), axis=0)
            interpolated_ctrl      = self.concatenate(interpolated_ctrl, ctrl_d.reshape(-1, 1))
        return interpolated_ctrl


    def __interpolate_until_endpoint(self, theta_current, theta_target):
        return np.linspace(
            start    = theta_current,
            stop     = theta_target,
            num      = (self.num_interpolation - self.endpoint_margin_step),
            endpoint = True,
        )


    def __interpolate_endpoint_margin(self, theta_target):
        return np.linspace(
            start    = theta_target,
            stop     = theta_target,
            num      = self.endpoint_margin_step,
            endpoint = True,
        )
