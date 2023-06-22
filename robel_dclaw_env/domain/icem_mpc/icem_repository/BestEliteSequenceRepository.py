import os
import pickle


class BestEliteSequenceRepository:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir


    def save(self,
            best_elite_action_sequence,
            best_elite_sample_sequence,
            best_object_state_sequence,
        ):
        save_path = os.path.join(self.save_dir, "best_elite_sequence.pkl")
        with open(save_path, "wb") as tf:
            pickle.dump(
                file = tf,
                obj  = {
                    "best_elite_action_sequence" : best_elite_action_sequence,
                    "best_elite_sample_sequence" : best_elite_sample_sequence,
                    "best_object_state_sequence" : best_object_state_sequence,
                }
            )


    def load(self):
        load_path = os.path.join(self.save_dir, "best_elite_sequence.pkl")
        # import ipdb; ipdb.set_trace()
        with open(load_path, "rb") as tf:
            best_elite_sequence = pickle.load(tf)
        return {
            "task_space_abs_position"          : best_elite_sequence["best_elite_action_sequence"],
            "task_space_differential_position" : best_elite_sequence["best_elite_sample_sequence"],
            "best_object_state_sequence"       : best_elite_sequence["best_object_state_sequence"],
        }

