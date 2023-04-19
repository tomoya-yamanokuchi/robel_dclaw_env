import hydra
from omegaconf import DictConfig
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from custom_service import create_feedable_ctrl_from_less_dim_ctrl
from CollectionfromRandomMotion import CollectionfromRandomMotion

'''
test dataset
'''

if __name__ == '__main__':
    @hydra.main(version_base=None, config_path="../../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):

        config.env.camera.z_distance = 0.4

        demo = CollectionfromRandomMotion(config)

        # << --- random action for each claw --- >>
        num_sample_clawX = 30; colored_noise_exponent_clawX=[1.0, 2.0, 3.0]
        random_sample = demo.get_random_ctrl(num_sample_clawX, colored_noise_exponent_clawX)

        ctrl_random_claw1 = create_feedable_ctrl_from_less_dim_ctrl(
            dim_totoal_ctrl=3, task_space_differential_ctrl=random_sample[:, :, :1], dimension_of_interst=[0])

        ctrl_random_claw2 = create_feedable_ctrl_from_less_dim_ctrl(
            dim_totoal_ctrl=3, task_space_differential_ctrl=random_sample[:, :, 1:2], dimension_of_interst=[1])

        ctrl_random_claw3 = create_feedable_ctrl_from_less_dim_ctrl(
            dim_totoal_ctrl=3, task_space_differential_ctrl=random_sample[:, :, 2:], dimension_of_interst=[2])

        # << --- random action for all claw --- >>
        num_sample_all_claw = 60;  colored_noise_exponent_all_claw=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ctrl_random_all = demo.get_random_ctrl(num_sample_all_claw, colored_noise_exponent_all_claw)

        demo.run_forward_model(ctrl_random_claw1, dataset_name =    "random_action_claw1_NumSample{}_NumColoredNoiseExponent{}".format(num_sample_clawX,    len(colored_noise_exponent_clawX)))
        demo.run_forward_model(ctrl_random_claw2, dataset_name =    "random_action_claw2_NumSample{}_NumColoredNoiseExponent{}".format(num_sample_clawX,    len(colored_noise_exponent_clawX)))
        demo.run_forward_model(ctrl_random_claw3, dataset_name =    "random_action_claw3_NumSample{}_NumColoredNoiseExponent{}".format(num_sample_clawX,    len(colored_noise_exponent_clawX)))
        demo.run_forward_model(ctrl_random_all  , dataset_name = "random_action_all_claw_NumSample{}_NumColoredNoiseExponent{}".format(num_sample_all_claw, len(colored_noise_exponent_all_claw)))
    main()
