import pathlib


def make_dir(*args):
    p = pathlib.Path(*args)
    p.mkdir(parents=True, exist_ok=True)
