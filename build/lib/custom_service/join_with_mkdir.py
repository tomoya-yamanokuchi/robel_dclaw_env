import os
import pathlib


def join_with_mkdir(*x, is_end_file=True):
    save_path = os.path.join(*x)

    if is_end_file: save_dir  = "/".join(save_path.split("/")[:-1])
    else          : save_dir  = save_path

    # file_name = save_path.split("/")[-1]
    path      = pathlib.Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    # import ipdb; ipdb.set_trace()
    return save_path
