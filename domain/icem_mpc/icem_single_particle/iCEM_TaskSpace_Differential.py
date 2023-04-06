import os
import copy
import time
import numpy as np
from typing import Any, List
from typing import Any, List, Tuple
from .CostHistory import CostHistory
from .EliteSetQueue import EliteSetQueue
from .visualization.VisualizationCollection import VisualizationCollection
from .ColoredNoiseSampler import ColoredNoiseSampler
from .ActionDimensionOfInterest import ActionDimensionOfInterest as ActionDoI
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from custom_service import concat



class iCEM_TaskSpace_Differential:
    def __init__(self,
            forward_model                : Any,
            forward_model_progress_check : Any,
            cost_function                : Any,
            dimension_of_interst         : List[int],
            num_sample                   : int,
            num_elite                    : int,
            planning_horizon             : int,
            dim_action                   : int,
            dim_path                     : int,
            num_cem_iter                 : int,
            decay_sample                 : float,
            colored_noise_exponent       : float,
            fraction_rate_elite          : float,
            lower_bound_sampling         : float,
            upper_bound_sampling         : float,
            TaskSpace                    : Any,
            lower_bound_simulated_path   : float,
            upper_bound_simulated_path   : float,
            alpha                        : float,
            init_std                     : float ,
            verbose                      : bool,
            verbose_additional           : bool  = False,
            is_verbose_newline           : bool  = False,
            save_visualization_dir       : str   = None,
            debug                        : bool  = False,
            figsize_path                 : Tuple = (7, 4),
            figsize_cost                 : Tuple = (5, 5),
        ):
        self.forward_model                = forward_model
        self.forward_model_progress_check = forward_model_progress_check
        self.cost_function                = cost_function
        self.dimension_of_interst         = dimension_of_interst
        self.num_sample                   = num_sample
        self.num_elite                    = num_elite
        self.planning_horizon             = planning_horizon
        self.dim_action                   = dim_action
        self.num_cem_iter                 = num_cem_iter
        self.decay_sample                 = decay_sample
        self.lower_bound_sampling         = lower_bound_sampling
        self.upper_bound_sampling         = upper_bound_sampling
        self.TaskSpace                    = TaskSpace
        self.alpha                        = alpha
        self.verbose                      = verbose
        self.verbose_additional           = verbose_additional
        self.is_verbose_newline           = is_verbose_newline
        self.save_visualization_dir       = save_visualization_dir
        self.init_std                     = init_std
        self.debug                        = debug
        self.colored_noise_exponent       = colored_noise_exponent

        self.elite_set_queue = EliteSetQueue(
            num_elite     = self.num_elite,
            fraction_rate = fraction_rate_elite,
        )

        self.colored_noise_sampler = ColoredNoiseSampler(
            planning_horizon = self.planning_horizon,
            dim_action       = self.dim_action,
        )

        self.time_now       = str(time.time())
        save_dir            = os.path.join(save_visualization_dir, self.time_now)
        self.vis_collection = VisualizationCollection()
        self.vis_collection.append("cost", save_dir, figsize_cost)
        self.vis_collection.append("simulated_paths", dim_path, planning_horizon, save_dir, lower_bound_simulated_path, upper_bound_simulated_path, figsize_path)
        self.vis_collection.append("sample", dim_action, figsize_path, save_dir, lower_bound_sampling, upper_bound_sampling)


        self.cost_history = CostHistory()

        self.iter_outer_loop = None


    def reset(self):
        if self.iter_outer_loop is None:
            self.iter_outer_loop = 0
        else: self.iter_outer_loop += 1
        self.mean = self._get_init_mean()
        self.std  = self._get_init_std()
        self.total_sample_size_in_optimze = 0
        self.total_proc_time_in_optimze   = 0


    def _get_init_mean(self):
        if self.iter_outer_loop > 0:
            # Shift mean time-wise
            if self.verbose: print("<< shit mean iter_outer{} >> \n".format(self.iter_outer_loop))
            shifted_mean  = copy.deepcopy(self.mean[:, 1:])
            last_new_mean = (self.lower_bound_sampling + self.upper_bound_sampling) / 2.0
            last_new_mean += np.zeros([1, 1, self.dim_action])
            return np.concatenate((shifted_mean, last_new_mean), axis=1)
        init_mean = (self.lower_bound_sampling + self.upper_bound_sampling) / 2.0
        init_mean = init_mean + np.zeros([self.planning_horizon, self.dim_action])
        return init_mean[np.newaxis, :, :]


    def _get_init_std(self):
        std_init = (self.upper_bound_sampling - self.lower_bound_sampling) / 2.0 * self.init_std
        std_init = std_init + np.zeros([self.planning_horizon, self.dim_action])
        return std_init[np.newaxis, :, :]


    def _decay_population_size(self, iter_inner_loop):
        minimum_sample = self.num_elite * 2
        decayed_sample = self.num_sample / (self.decay_sample**iter_inner_loop)
        num_sample     = max(minimum_sample, int(decayed_sample))
        if self.verbose:
            print("[iCEM] iter_outer={} | iter_inner={}/{} | decayed_sample_size={: 4}".format(
                self.iter_outer_loop, iter_inner_loop, self.num_cem_iter-1, num_sample), end=' | ')
        return num_sample


    def _sample(self, num_sample):
        colored_noise = None
        indexes       = np.array_split(range(num_sample), len(self.colored_noise_exponent))
        for i, beta in enumerate(self.colored_noise_exponent):
            num_sample_i  = indexes[i].shape[0]
            noise_beta    = self.colored_noise_sampler.sample(num_sample_i, beta=beta)
            colored_noise = concat(colored_noise, noise_beta, axis=0)
        return self.__clip_samples((colored_noise * self.std) + self.mean)


    def __clip_samples(self, x):
        return np.clip(a=x, a_min=self.lower_bound_sampling, a_max=self.upper_bound_sampling)


    def _add_fraction_of_elite_set(self, samples, iter_inner_loop):
        if self.elite_set_queue.is_empty():
            return samples
        if iter_inner_loop < self.num_cem_iter - 1:
            return np.concatenate([samples, self.elite_set_queue.get_elites()], axis=0)
        if self.verbose_additional:
            print(" --> add_fraction_of_elite_set (iter {})".format(iter_inner_loop), end=' | ')
        shited_elite_sample = self.elite_set_queue.get_shifted_elites()
        last_action         = self._sample(self.elite_set_queue.num_reuse)[:, -1:]
        elite_samples       = np.concatenate((shited_elite_sample, last_action), axis=1)
        return np.concatenate([samples, elite_samples], axis=0)


    def _add_minmaxmean_action_sample(self, samples):
        num_sample, step, dim = samples.shape
        sampling_mean         = (self.upper_bound_sampling + self.lower_bound_sampling) * 0.5
        mean_action_sample    = np.zeros([1, step, dim]) + sampling_mean
        minimum_action_sample = np.zeros([1, step, dim]) + self.lower_bound_sampling
        maximum_action_sample = np.zeros([1, step, dim]) + self.upper_bound_sampling
        return np.concatenate([samples, mean_action_sample, minimum_action_sample, maximum_action_sample], axis=0)


    def _add_mean_action_at_last_iteration(self, samples, iter_inner_loop):
        if self.num_cem_iter == 1                  : return samples
        if iter_inner_loop != self.num_cem_iter - 1: return samples
        if self.verbose_additional:
            print(" --> add_mean_action_at_last_iteration (iter {})".format(iter_inner_loop), end=' | ')
        return np.concatenate([samples, copy.deepcopy(self.mean)], axis=0)


    def _update_distributions(self, elite_set):
        new_mean  = np.mean(elite_set, axis=0, keepdims=True)
        new_std   = np.std( elite_set, axis=0, keepdims=True)
        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean
        self.std  = (1 - self.alpha) * new_std  + self.alpha * self.std


    def _get_index_elite(self, cost):
        index_elite = np.argsort(np.array(cost))[:self.num_elite]
        if self.verbose: print("min cost={:.3f}".format(cost[index_elite][0]), end=' | ')
        return index_elite


    def optimize(self, constant_setting, action_bias, target):
        action_doi = ActionDoI(action_bias, self.dimension_of_interst)
        for i in range(self.num_cem_iter):
            time_start        = time.time()
            num_sample_i      = self._decay_population_size(i)
            samples           = self._sample(num_sample_i)
            samples           = self._add_minmaxmean_action_sample(samples)
            samples           = self._add_fraction_of_elite_set(samples, i)
            samples           = self._add_mean_action_at_last_iteration(samples, i)
            num_samples       = samples.shape[0]
            self.total_sample_size_in_optimze += num_samples
            if self.verbose: print("total_sample_size={: 4}".format(num_samples), end=' | ')
            forward_results   = self._forward(constant_setting, samples)
            cost              = self.cost_function(forward_results=forward_results, target=target)
            assert cost.shape == (num_samples,), print("{} != {}".format(cost.shape, (num_samples,)))
            index_elite       = self._get_index_elite(cost)
            best_elite_sample = samples[index_elite[:1]]
            forward_results_progress = self._forward_progress_check(constant_setting, best_elite_sample, i, target)
            elites            = copy.deepcopy(samples[index_elite])
            self.elite_set_queue.append(elites)
            self._update_distributions(elites)
            self.cost_history.append(cost)
            # ---- visualize ----
            if self.save_visualization_dir is None: continue
            self.vis_collection.plot("cost"  ,              cost, i, self.iter_outer_loop, num_sample_i)
            self.vis_collection.plot("simulated_paths",     forward_results, index_elite, target, i, self.iter_outer_loop, num_sample_i)
            self.vis_collection.plot("sample",              samples, samples[index_elite], i, self.iter_outer_loop, num_sample_i)
            # ---- time count ----
            self.update_total_process_time(time_start)
        if self.verbose: self._print_optimize_info()
        return {
            "cost"              : cost,
            "state"             : forward_results_progress["state"],
            "best_elite_action" : forward_results_progress["task_space_ctrl"],
            "best_elite_sample" : best_elite_sample[0, 0],
        }


    def _forward(self, constant_setting, actions):
        if self.debug: actions = actions[:1]
        multiproc = ForwardModelMultiprocessing(verbose=False)
        results, proc_time = multiproc.run(
            rollout_function = self.forward_model,
            constant_setting = constant_setting,
            ctrl             = actions,
        )
        if self.verbose_additional: print("time={:.3f}".format(proc_time), end=' | ')
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
                    "save_fig_dir"    : os.path.join(self.save_visualization_dir, self.time_now),
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
        if self.verbose:
            if self.is_verbose_newline: print("time={:.3f}".format(elapsed_time))
            else: print("time={:.3f}".format(elapsed_time), end=' | ')


    def _print_optimize_info(self):
        print("\n")
        print("------------------------------------------------")
        print("    total_sample_size_in_optimze : {}           ".format(self.total_sample_size_in_optimze))
        print("    total_proc_time_in_optimze   : {:.2f} [sec]".format(self.total_proc_time_in_optimze))
        print("------------------------------------------------")
