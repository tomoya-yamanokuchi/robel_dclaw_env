import hydra
from omegaconf import DictConfig
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.custom_service import create_feedable_ctrl_from_less_dim_ctrl
from CollectionfromRandomMotion import CollectionfromRandomMotion

'''
test dataset
'''

if __name__ == '__main__':
    @hydra.main(version_base=None, config_path="../../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):

        config.env.camera.z_distance = 0.4 # ロボットがフレームアウトする情報欠損を起こさないようにカメラを引きで設定
        config.env.init_state.task_space_position = [0.14, 0.14, 0.14] # 指の初期位置を中心に？

        demo = CollectionfromRandomMotion(config)

        num_sample_all_claw             = 600
        colored_noise_exponent_all_claw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ctrl_random_all                 = demo.get_random_ctrl(num_sample_all_claw, colored_noise_exponent_all_claw)

        demo.run_forward_model(
            ctrl_random_all,
            "random_action_all_claw_NumSample{}_NumColoredNoiseExponent{}".format(num_sample_all_claw, len(colored_noise_exponent_all_claw))
        )
    main()
