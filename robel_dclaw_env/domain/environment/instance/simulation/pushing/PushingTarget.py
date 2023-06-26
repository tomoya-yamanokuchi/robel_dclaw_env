from transforms3d.euler import euler2quat, quat2euler



class PushingTarget:
    def __init__(self, sim):
        self.sim               = sim
        self.target_bid_left   = sim.model.body_name2id('target_left_mark')
        self.target_bid_center = sim.model.body_name2id('target_center_mark')
        self.target_bid_right  = sim.model.body_name2id('target_right_mark')


    def set_target_visible(self, target_visible):
        target_sid_list = [
            self.sim.model.site_name2id('target_left_mark'),
            self.sim.model.site_name2id('target_center_mark'),
            self.sim.model.site_name2id('target_right_mark'),
        ]
        if target_visible:
            for target_sid in target_sid_list:
                self.sim.model.site_rgba[target_sid] = [0.0,  0.92156863, 0.0, 1]; return
        self.sim.model.site_rgba[target_sid] = [0.0, 0.0, 0.0, 0.0]


    def set_target_position(self, target_position):
        self.sim.model.body_quat[self.target_bid] = euler2quat(0, 0, float(target_position))
