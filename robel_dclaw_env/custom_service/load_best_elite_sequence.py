import pickle


def load_best_elite_sequence(load_path: str):
    with open(load_path, "rb") as tf:
        best_elite_sequence              = pickle.load(tf)
        task_space_differential_position = best_elite_sequence["best_elite_sample_sequence"]
        task_space_abs_position          = best_elite_sequence["best_elite_action_sequence"]
        best_object_state_sequence       = best_elite_sequence["best_object_state_sequence"]
    return {
        "task_space_differential_position" : task_space_differential_position,
        "task_space_abs_position"          : task_space_abs_position,
        "best_object_state_sequence"       : best_object_state_sequence,
    }
