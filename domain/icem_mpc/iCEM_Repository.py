import pickle
from custom_service import join_with_mkdir, time_as_string


class iCEM_Repository:
    def __init__(self, config_icem):
        self.config_icem = config_icem


    def save(self,
            best_elite_action_sequence,
            best_elite_sample_sequence,
            best_object_state_sequence
        ):
        save_path = self._get_save_path()
        with open(save_path, "wb") as tf:
            pickle.dump(
                file = tf,
                obj  = {
                    "best_elite_action_sequence" : best_elite_action_sequence,
                    "best_elite_sample_sequence" : best_elite_sample_sequence,
                    "best_object_state_sequence" : best_object_state_sequence,
                }
            )


    def _get_save_path(self):
        return join_with_mkdir("./", "best_elite_sequence",
            "best_elite_sequence-[num_cem_iter={}]-[planning_horizon={}]-[num_sample={}]-{}.pkl".format(
                self.config_icem.num_cem_iter,
                self.config_icem.planning_horizon,
                self.config_icem.num_sample,
                time_as_string(),
            )
        )
