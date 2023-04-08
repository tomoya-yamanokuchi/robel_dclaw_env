import os, time, copy, pathlib
from omegaconf import OmegaConf
from .BestEliteSequenceRepository import BestEliteSequenceRepository
from .FigureRepository import FigureRepository
from .ConfigRepository import ConfigRepository
from custom_service import join_with_mkdir


class iCEM_Repository:
    def __init__(self):
        self.config = None


    def set_config_and_repository(self, config):
        assert self.config is None
        self.config = config
        self._set_repositories()


    def load_config_and_repository(self, dir_name):
        assert self.config is None
        p           = pathlib.Path(".")
        self.config = OmegaConf.load(
            os.path.join(p.absolute(), "saved_data_icem", dir_name, "config.yaml")
        )
        self._set_repositories()
        return copy.deepcopy(self.config)


    def _set_repositories(self):
        save_dir_options = {
            "create" : self._create_save_dir_name,
            "copy"   : self.config.save_dir,
        }
        self.save_dir = save_dir_options["create"]() if self.config.save_dir is None else save_dir_options["copy"]
        self.best_elite_sequence_repository = BestEliteSequenceRepository(self.save_dir)
        self.figure_repository              = FigureRepository(self.save_dir)
        self.config_repository              = ConfigRepository(self.save_dir)


    def _create_save_dir_name(self):
        return join_with_mkdir(".",
            "saved_data_icem",
            "[num_sample={}]-[num_subparticle={}]-[num_cem_iter={}]-[colored_noise_exponent={}]-{}".format(
                    self.config.icem.num_sample,
                    self.config.icem.num_subparticle,
                    self.config.icem.num_cem_iter,
                    self.config.icem.colored_noise_exponent,
                    time.time()
                ),
            is_end_file=False
        )


    def save_best_elite_sequence(self, **kwargs):
        self.best_elite_sequence_repository.save(**kwargs)

    def load_best_elite_sequence(self):
        return self.best_elite_sequence_repository.load()


    def save_figure(self, **kwargs):
        self.figure_repository.save(**kwargs)

    def save_config(self):
        self.config_repository.save(self.config)
