from .elements.ValveVisualization import ValveVisualization
from .elements.SampleVisualizatioin import SampleVisualizatioin
from .elements.ActionVisualizatioin import ActionVisualizatioin
from .elements.CostVisualization import CostVisualization
from .elements.PerturbatedNominalVisualizatioin import PerturbatedNominalVisualizatioin
from .elements.SubparticleValveVisualization import SubparticleValveVisualization
from .elements.UnitSubparticleSampleVisualization import UnitSubparticleSampleVisualization
from .elements.TotalSubparticleSampleVisualization import TotalSubparticleSampleVisualization


class VisualizationCollection:
    def __init__(self):
        self.collection = {}


    def append(self, name, *args):
        candidates = {
            "cost"                        : CostVisualization,
            "simulated_paths"             : ValveVisualization,
            "sample"                      : SampleVisualizatioin,
            "action"                      : ActionVisualizatioin,
            "perturbated_nominal"         : PerturbatedNominalVisualizatioin,
            "subparticle_simulated_paths" : SubparticleValveVisualization,
            "unit_subparticle_sample"     : UnitSubparticleSampleVisualization,
            "total_subparticle_sample"    : TotalSubparticleSampleVisualization,
        }

        self.collection[name] = candidates[name](*args)


    def plot(self, name, *args):
        self.collection[name].plot(*args)


    # def names(self):
        # return
