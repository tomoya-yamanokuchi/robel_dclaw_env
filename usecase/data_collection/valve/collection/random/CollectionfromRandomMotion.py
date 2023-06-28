import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from usecase.data_collection.rollout.rollout_dataset_collection_with_differential_ctrl import rollout_dataset_collection_with_differential_ctrl
from robel_dclaw_env.domain.environment.EnvironmentFactory import EnvironmentFactory
from robel_dclaw_env.domain.forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from robel_dclaw_env.domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from robel_dclaw_env.domain.environment.task_space.manifold_1d.TaskSpaceDifferentialPositionValue_1D_Manifold import TaskSpaceDifferentialPositionValue_1D_Manifold
from domain.icem_mpc.icem_mpc.population.PopulationSampler import PopulationSampler
from domain.icem_mpc.icem_mpc.population.PopulationSampingDistribution import PopulationSampingDistribution
from domain.icem_mpc.icem_mpc.population.PopulationSampler import PopulationSampler
from robel_dclaw_env.custom_service import time_as_string, create_feedable_ctrl_from_less_dim_ctrl



class CollectionfromRandomMotion:
    def __init__(self, config):
        self.config = config


    def get_random_ctrl(self, num_sample, colored_noise_exponent):
        population_sampling_dist = PopulationSampingDistribution(self.config.icem)
        population_sampling_dist.reset_init_distribution(iter_outer_loop=0)
        population_sampler = PopulationSampler(self.config.icem)
        random_ctrl        = population_sampler.sample(
            mean                   = population_sampling_dist.mean,
            std                    = population_sampling_dist.std,
            colored_noise_exponent = colored_noise_exponent,
            num_sample             = num_sample,
        )
        return random_ctrl



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
                "TaskSpaceAbs" : TaskSpacePositionValue_1D_Manifold,
                "TaskSpaceDiff": TaskSpaceDifferentialPositionValue_1D_Manifold,
                "dataset_name" : dataset_name + "_" + time_as_string(),
            },
            ctrl = ctrl,
        )



