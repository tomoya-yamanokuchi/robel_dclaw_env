import copy
import hydra
from omegaconf import DictConfig
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from CollectionfromNominalwithNoise import CollectionfromNominalwithNoise
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        repository   = Repository(**config.nominal, read_only= True)
        repository.open(filename="nominal")
        nominal_ctrl = repository.repository["ctrl"]["task_space_diff_position"]
        repository.close()

        config.env.camera.z_distance = 0.4
        demo = CollectionfromNominalwithNoise(config)

        # << --- random action for each claw --- >>
        num_sample = 10; colored_noise_exponent = [1.0, 2.0, 3.0]; planning_horizon = nominal_ctrl.shape[0]
        nominal_with_noise0 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.10, sampling_bound_width = 0.01)
        nominal_with_noise1 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.15, sampling_bound_width = 0.02)
        nominal_with_noise2 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.20, sampling_bound_width = 0.03)
        nominal_with_noise3 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.25, sampling_bound_width = 0.04)
        nominal_with_noise4 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.30, sampling_bound_width = 0.05)
        # nominal_with_noise1 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.50, sampling_bound_width = 0.025)
        # nominal_with_noise2 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.55, sampling_bound_width = 0.050)
        # nominal_with_noise3 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.60, sampling_bound_width = 0.075)
        # nominal_with_noise4 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.65, sampling_bound_width = 0.100)
        # nominal_with_noise5 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.75, sampling_bound_width = 0.200)


        # import ipdb; ipdb.set_trace()
        demo.run_forward_model(nominal_with_noise0, dataset_name="nominal_with_noise0_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        demo.run_forward_model(nominal_with_noise1, dataset_name="nominal_with_noise1_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        demo.run_forward_model(nominal_with_noise2, dataset_name="nominal_with_noise2_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        demo.run_forward_model(nominal_with_noise3, dataset_name="nominal_with_noise3_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        demo.run_forward_model(nominal_with_noise4, dataset_name="nominal_with_noise4_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        # demo.run_forward_model(nominal_with_noise5, dataset_name="nominal_with_noise5_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))

    main()
