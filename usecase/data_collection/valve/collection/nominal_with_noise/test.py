import hydra
from omegaconf import DictConfig
from CollectionfromNominalwithNoise import CollectionfromNominalwithNoise

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):

        config.env.camera.z_distance = 0.4

        demo = CollectionfromNominalwithNoise(config)
        nominal_ctrl = demo.load_nominal_ctrl(
            "[num_sample=500]-[num_subparticle=10]-[num_cem_iter=7]-[colored_noise_exponent=[0.5, 1.0, 2.0, 3.0, 4.0]]-1680989914.145154"
        )

        # << --- random action for each claw --- >>
        num_sample = 10; colored_noise_exponent = [1.0, 2.0, 3.0]; planning_horizon = nominal_ctrl.shape[0]
        nominal_with_noise1 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.50, sampling_bound_width = 0.025)
        nominal_with_noise2 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.55, sampling_bound_width = 0.050)
        nominal_with_noise3 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.60, sampling_bound_width = 0.075)
        nominal_with_noise4 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.65, sampling_bound_width = 0.100)
        nominal_with_noise5 = demo.add_random_noise_to_nominal(nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std = 0.75, sampling_bound_width = 0.200)


        # import ipdb; ipdb.set_trace()
        demo.run_forward_model(nominal_with_noise1, dataset_name="nominal_with_noise1_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        demo.run_forward_model(nominal_with_noise2, dataset_name="nominal_with_noise2_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        demo.run_forward_model(nominal_with_noise3, dataset_name="nominal_with_noise3_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        demo.run_forward_model(nominal_with_noise4, dataset_name="nominal_with_noise4_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))
        demo.run_forward_model(nominal_with_noise5, dataset_name="nominal_with_noise5_NumSample{}_NumColoredNoiseExponent{}".format(num_sample, len(colored_noise_exponent)))

    main()
