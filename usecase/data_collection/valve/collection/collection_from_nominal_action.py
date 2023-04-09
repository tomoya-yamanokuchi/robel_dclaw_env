import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from usecase.data_collection.rollout.rollout_dataset_collection_with_differential_ctrl import rollout_dataset_collection_with_differential_ctrl
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from domain.icem_mpc.icem_mpc.population.PopulationSampler import PopulationSampler
from domain.icem_mpc.icem_mpc.population.PopulationSampingDistribution import PopulationSampingDistribution
from domain.icem_mpc.icem_mpc.population.PopulationSampler import PopulationSampler
from custom_service import time_as_string, create_feedable_ctrl_from_less_dim_ctrl
from domain.icem_mpc.icem_repository.iCEM_Repository import iCEM_Repository
from domain.icem_mpc.icem_mpc.visualization.VisualizationCollection import VisualizationCollection
from save_ctrl_sample_figure import save_path


class CollectionfromNominalAction:
    def __init__(self, config):
        self.config = config


    def load_nominal_ctrl(self, load_path):
        self.icem_repository       = iCEM_Repository()
        config                     = self.icem_repository.load_config_and_repository(load_path)
        result_best_elite_sequence = self.icem_repository.load_best_elite_sequence()
        return result_best_elite_sequence["task_space_differential_position"]


    def add_random_noise_to_nominal(self, nominal_ctrl, num_sample, colored_noise_exponent, planning_horizon, init_std, sampling_bound_width):
        self.config.icem.lower_bound_sampling = -sampling_bound_width
        self.config.icem.upper_bound_sampling =  sampling_bound_width
        self.config.icem.init_std             = init_std
        self.config.icem.planning_horizon     = planning_horizon
        population_sampling_dist = PopulationSampingDistribution(self.config.icem)
        population_sampling_dist.reset_init_distribution(iter_outer_loop=0)
        population_sampler = PopulationSampler(self.config.icem)
        random_noise       = population_sampler.sample(
            mean                   = population_sampling_dist.mean,
            std                    = population_sampling_dist.std,
            colored_noise_exponent = colored_noise_exponent,
            num_sample             = num_sample,
        )
        save_path(random_noise, self.icem_repository.save_dir, figsize=(4, 7), figname="random_noise_init_std={}".format(init_std))
        # ------
        nominal_with_noise = nominal_ctrl[np.newaxis,:,:] + random_noise
        save_path(nominal_with_noise, self.icem_repository.save_dir, figsize=(4, 7), figname="nominal_with_noise={}".format(init_std))
        return nominal_with_noise



    def run_forward_model(self, ctrl, dataset_name):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=self.config.env.env_name)
        init_state = state_subclass(**self.config.env.init_state)
        multiproc          = ForwardModelMultiprocessing(verbose=False)
        results, proc_time = multiproc.run(
            rollout_function = rollout_dataset_collection_with_differential_ctrl,
            constant_setting = {
                "env_subclass" : env_subclass,
                "config"       : self.config,
                "init_state"   : init_state,
                "TaskSpace"    : TaskSpacePositionValue_1D_Manifold,
                "dataset_name" : dataset_name + "_" + time_as_string(),
            },
            ctrl = ctrl,
        )



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = CollectionfromNominalAction(config)
        nominal_ctrl = demo.load_nominal_ctrl(
            "[num_sample=500]-[num_subparticle=10]-[num_cem_iter=7]-[colored_noise_exponent=[0.5, 1.0, 2.0, 3.0, 4.0]]-1680989914.145154"
        )

        # << --- random action for each claw --- >>
        num_sample = 100; colored_noise_exponent = [1.0, 2.0, 3.0]; planning_horizon = nominal_ctrl.shape[0]
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
