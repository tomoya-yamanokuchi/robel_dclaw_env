import os
import copy
import time
import numpy as np
from typing import Any, List
from typing import Any, List, Tuple
from ..CostHistory import CostHistory
from ..EliteSetQueue import EliteSetQueue
from ..visualization.VisualizationCollection import VisualizationCollection
from ..ColoredNoiseSampler import ColoredNoiseSampler
from ..ActionDimensionOfInterest import ActionDimensionOfInterest as ActionDoI
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from custom_service import concat


from .PopulationSampingDistribution import PopulationSampingDistribution
from .PopulationSizeScheduler import PopulationSizeScheduler



class iCEM_Subparticle:
    def __init__(self,
            forward_model                : Any,
            forward_model_progress_check : Any,
            cost_function                : Any,
            repository                   : Any,
            config,
        ):
        self.forward_model                = forward_model
        self.forward_model_progress_check = forward_model_progress_check
        self.cost_function                = cost_function
        self.config                       = config

        self.elite_set_queue = EliteSetQueue(
            num_elite     = config.num_elite,
            fraction_rate = config.fraction_rate_elite,
        )

        self.colored_noise_sampler = ColoredNoiseSampler(
            planning_horizon = self.config.planning_horizon,
            dim_action       = self.config.dim_action,
        )

        self.population_sampling_dist  = PopulationSampingDistribution(config)
        self.population_size_scheduler = PopulationSizeScheduler(config)

        self.time_now       = str(time.time())
        self.vis_collection = VisualizationCollection()
        self.vis_collection.append("cost",                        repository)
        self.vis_collection.append("subparticle_simulated_paths", repository)
        self.vis_collection.append("sample",                      repository)
        self.vis_collection.append("unit_subparticle_sample",     repository)
        self.vis_collection.append("total_subparticle_sample",    repository)

        self.cost_history    = CostHistory()
        self.iter_outer_loop = None


    def reset(self):
        if self.iter_outer_loop is None:
            self.iter_outer_loop = 0
        else: self.iter_outer_loop += 1
        self.population_sampling_dist.reset_init_distribution(self.iter_outer_loop)
        self.total_sample_size_in_optimze = 0
        self.total_proc_time_in_optimze   = 0


    def print_info(self, iter_inner_loop, num_sample):
        if self.config.verbose:
            print("[iCEM] iter_outer={} | iter_inner={}/{} | decayed_sample_size={: 4}".format(
                self.iter_outer_loop, iter_inner_loop, self.config.num_cem_iter-1, num_sample), end=' | ')



    def _sample(self, num_sample):
        colored_noise = None
        indexes       = np.array_split(range(num_sample), len(self.config.colored_noise_exponent))
        for i, beta in enumerate(self.config.colored_noise_exponent):
            num_sample_i  = indexes[i].shape[0]
            noise_beta    = self.colored_noise_sampler.sample(num_sample_i, beta=beta)
            colored_noise = concat(colored_noise, noise_beta, axis=0)
        # ------
        mean = self.population_sampling_dist.mean
        std  = self.population_sampling_dist.std
        return self.__clip_samples((colored_noise * std) + mean)


    def __clip_samples(self, x):
        return np.clip(a=x, a_min=self.config.lower_bound_sampling, a_max=self.config.upper_bound_sampling)


    def _add_fraction_of_elite_set(self, samples, iter_inner_loop):
        if self.elite_set_queue.is_empty():
            return samples
        if iter_inner_loop < self.config.num_cem_iter - 1:
            return np.concatenate([samples, self.elite_set_queue.get_elites()], axis=0)
        if self.config.verbose_additional:
            print(" --> add_fraction_of_elite_set (iter {})".format(iter_inner_loop), end=' | ')
        shited_elite_sample = self.elite_set_queue.get_shifted_elites()
        last_action         = self._sample(self.elite_set_queue.num_reuse)[:, -1:]
        elite_samples       = np.concatenate((shited_elite_sample, last_action), axis=1)
        return np.concatenate([samples, elite_samples], axis=0)


    def _add_minmaxmean_action_sample(self, samples):
        num_sample, step, dim = samples.shape
        sampling_mean         = (self.config.upper_bound_sampling + self.config.lower_bound_sampling) * 0.5
        mean_action_sample    = np.zeros([1, step, dim]) + sampling_mean
        minimum_action_sample = np.zeros([1, step, dim]) + self.config.lower_bound_sampling
        maximum_action_sample = np.zeros([1, step, dim]) + self.config.upper_bound_sampling
        return np.concatenate([samples, mean_action_sample, minimum_action_sample, maximum_action_sample], axis=0)


    def _add_mean_action_at_last_iteration(self, samples, iter_inner_loop):
        if self.config.num_cem_iter == 1                  : return samples
        if iter_inner_loop != self.config.num_cem_iter - 1: return samples
        if self.config.verbose_additional:
            print(" --> add_mean_action_at_last_iteration (iter {})".format(iter_inner_loop), end=' | ')
        return np.concatenate([samples, copy.deepcopy(self.population_sampling_dist.mean)], axis=0)



    def _get_index_elite(self, cost):
        index_elite = np.argsort(np.array(cost))[:self.config.num_elite]
        if self.config.verbose: print("min cost={:.3f} (submean={:.3f})".format(
            cost[index_elite][0], (cost[index_elite][0] / self.config.num_subparticle)), end=' | ')
        return index_elite


    def optimize(self, constant_setting, target):
        for i in range(self.config.num_cem_iter):
            time_start               = time.time()
            num_sample_i             = self.population_size_scheduler.decay(i); self.print_info(i, num_sample_i)
            samples                  = self._sample(num_sample_i)
            samples                  = self._add_minmaxmean_action_sample(samples)
            samples                  = self._add_fraction_of_elite_set(samples, i)
            samples                  = self._add_mean_action_at_last_iteration(samples, i)
            num_samples              = samples.shape[0]
            self.total_sample_size_in_optimze += num_samples
            if self.config.verbose: print("total_sample_size={: 4}".format(num_samples), end=' | ')
            # ----------------------------
            subparticle_group_list   = self._extend_as_subparticle(samples, i)
            perturbated_samples      = np.concatenate(subparticle_group_list)
            forward_results          = self._forward(constant_setting, perturbated_samples)
            cost                     = self.get_cost(forward_results, target, num_samples)
            assert cost.shape == (num_samples,), print("{} != {}".format(cost.shape, (num_samples,)))
            # ----------------------------
            index_elite              = self._get_index_elite(cost)
            best_elite_sample        = samples[index_elite[:1]]
            forward_results_progress = self._forward_progress_check(constant_setting, best_elite_sample, i, target)
            cost_elite               = self.get_cost(forward_results_progress, target, 1)
            elites                   = copy.deepcopy(samples[index_elite])
            self.elite_set_queue.append(elites)
            self.population_sampling_dist.update_distribution(elites)
            self.cost_history.append(cost)
            # ---- visualize ----
            if self.config.save_visualization_dir is None: continue
            self.vis_collection.plot("cost"  ,                      cost, i, self.iter_outer_loop, num_samples)
            self.vis_collection.plot("subparticle_simulated_paths", forward_results, index_elite, target, i, self.iter_outer_loop, num_samples)
            self.vis_collection.plot("sample",                      samples, samples[index_elite], i, self.iter_outer_loop, num_samples)
            self.vis_collection.plot("total_subparticle_sample",    subparticle_group_list, index_elite, i, self.iter_outer_loop, perturbated_samples.shape[0])
            # ---- time count ----
            # import ipdb; ipdb.set_trace()
            self.update_total_process_time(time_start)
        if self.config.verbose: self._print_optimize_info()
        return {
            "cost"              : cost,
            "state"             : forward_results_progress["state"],
            "best_elite_action" : forward_results_progress["task_space_ctrl"],
            "best_elite_sample" : best_elite_sample[0, 0],
        }


    def _extend_as_subparticle(self, samples, iter_inner_loop):
        noise = np.random.randn(self.config.num_subparticle, self.config.planning_horizon, self.config.dim_action) * self.config.std_subparticle
        num_samples = samples.shape[0]
        subparticle_group_list = []
        for i in range(num_samples):
            perturbated_samples = self.__clip_samples(samples[i][np.newaxis, :, :] + noise)
            subparticle_group_list.append(perturbated_samples)
            # self.vis_collection.plot("unit_subparticle_sample", perturbated_samples, i, iter_inner_loop, self.iter_outer_loop, self.config.num_subparticle)
        return subparticle_group_list


    def get_cost(self, forward_results, target, num_samples):
        cost_naive        = self.cost_function(forward_results=forward_results, target=target)
        cost_subparticles = np.array_split(cost_naive, indices_or_sections=num_samples, axis=0)
        cost_subparticles = np.stack(cost_subparticles, axis=0)
        cost              = np.sum(cost_subparticles, axis=-1)
        assert cost.shape == (num_samples,), print("{} != {}".format(cost.shape, (num_samples,)))
        if (self.config.verbose) and (num_samples==1): print("cost_elite={: .3f}".format(float(cost)), end=" | ")
        return cost


    def _forward(self, constant_setting, actions):
        if self.config.debug: actions = actions[:1]
        multiproc = ForwardModelMultiprocessing(verbose=False)
        results, proc_time = multiproc.run(
            rollout_function = self.forward_model,
            constant_setting = constant_setting,
            ctrl             = actions,
        )
        if self.config.verbose_additional: print("time={:.3f}".format(proc_time), end=' | ')
        return results



    def _forward_progress_check(self, constant_setting, best_elite_action, iter_inner_loop, target):
        assert best_elite_action.shape[0] == 1
        multiproc = ForwardModelMultiprocessing(verbose=False, result_aggregation=False)
        results, proc_time = multiproc.run(
            rollout_function = self.forward_model_progress_check,
            constant_setting = {
                **constant_setting,
                **{
                    "iter_outer_loop" : self.iter_outer_loop,
                    "iter_inner_loop" : iter_inner_loop,
                    "save_fig_dir"    : os.path.join(self.config.save_visualization_dir, self.time_now),
                    "dataset_name"    : self.time_now,
                    "target"          : target,
                },
            },
            ctrl = best_elite_action,
        )
        return results


    def update_total_process_time(self, time_start):
        elapsed_time = time.time() - time_start
        self.total_proc_time_in_optimze += elapsed_time
        if self.config.verbose:
            if self.config.is_verbose_newline: print("time={:.3f}".format(elapsed_time))
            else: print("time={:.3f}".format(elapsed_time), end=' | ')


    def _print_optimize_info(self):
        print("\n")
        print("------------------------------------------------")
        print("    total_sample_size_in_optimze : {}           ".format(self.total_sample_size_in_optimze))
        print("    total_proc_time_in_optimze   : {:.2f} [sec]".format(self.total_proc_time_in_optimze))
        print("------------------------------------------------")
