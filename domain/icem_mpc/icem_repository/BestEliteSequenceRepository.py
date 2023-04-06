import pickle
from custom_service import join_with_mkdir, time_as_string


class BestEliteSequenceRepository:
    def __init__(self, save_dir: str, config_icem):
        self.save_dir    = save_dir
        self.config_icem = config_icem


    def save(self,
            best_elite_action_sequence,
            best_elite_sample_sequence,
            best_object_state_sequence,
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
        return join_with_mkdir(self.save_dir, "best_elite_sequence",
            "best_elite_sequence-[num_cem_iter={}]-[planning_horizon={}]-[num_sample={}]-[nominal={}]-{}.pkl".format(
                self.config_icem.num_cem_iter,
                self.config_icem.planning_horizon,
                self.config_icem.num_sample,
                self.nominal,
                time_as_string(),
            )
        )
