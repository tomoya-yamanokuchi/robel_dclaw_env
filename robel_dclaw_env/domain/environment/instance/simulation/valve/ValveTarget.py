from transforms3d.euler import euler2quat, quat2euler



class ValveTarget:
    def __init__(self, sim):
        self.sim        = sim
        self.target_bid = sim.model.body_name2id('target')


    def set_target_visible(self, target_visible):
        target_sid = self.sim.model.site_name2id('tmark')
        if target_visible:
            self.sim.model.site_rgba[target_sid] = [0.0,  0.92156863, 0.0, 1]; return
        self.sim.model.site_rgba[target_sid] = [0.0, 0.0, 0.0, 0.0]


    def set_target_position(self, target_position):
        self.sim.model.body_quat[self.target_bid] = euler2quat(0, 0, float(target_position))
