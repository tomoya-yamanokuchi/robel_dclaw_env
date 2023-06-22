import os


class FigureRepository:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir


    def save(self, fig, fname):
        fig.savefig(
            fname = os.path.join(self.save_dir, fname),
            dpi   = 300
        )
