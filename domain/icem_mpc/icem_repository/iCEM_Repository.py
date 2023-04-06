import time
from .BestEliteSequenceRepository import BestEliteSequenceRepository
from .FigureRepository import FigureRepository
from .ConfigRepository import ConfigRepository
from custom_service import join_with_mkdir


class iCEM_Repository:
    def __init__(self, config):
        self.config                         = config
        self.save_dir                       = join_with_mkdir(".", "saved_data_icem", str(time.time()), is_end_file=False)
        self.best_elite_sequence_repository = BestEliteSequenceRepository(self.save_dir, config.icem)
        self.figure_repository              = FigureRepository(self.save_dir)
        self.config_repository              = ConfigRepository(self.save_dir)


    def save_best_elite_sequence(self, *args):
        self.best_elite_sequence_repository.save(*args)


    def save_figure(self, *args):
        self.figure_repository.save(*args)


    def save_config(self, *args):
        self.config_repository.save(*args)
