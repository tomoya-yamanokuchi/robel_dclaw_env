import cv2, time, copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from custom_service import print_info
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from custom_service import NTD, time_as_string
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository


class Demo_task_space:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        env        = env_subclass(config.env)
        init_state = state_subclass(**config.env.init_state)

        step         = 10
        dataset_name = time_as_string()
        repository   = Repository(dataset_dir="./dataset", dataset_name=dataset_name, read_only=False)

        for s in range(1):
            repository.open(filename='domain{}_action{}'.format(8888, s))
            image_list = []
            state_list = []
            ctrl_list  = []

            env.reset(init_state)
            state  = env.get_state()
            task_t = state.state['task_space_position'].value.squeeze()
            task_g = copy.deepcopy(task_t)
            # task_g -= 0.2

            for t in range(step):
                img   = env.render()
                state = env.get_state()
                env.view()

                task_g                       -= 0.05
                task_space_ctrl               = TaskSpacePositionValue_1D_Manifold(NTD(task_g))
                ctrl                          = env.set_ctrl_task_space(task_space_ctrl)
                ctrl.task_space_diff_position = task_g

                image_list.append(img)
                state_list.append(state)
                ctrl_list.append(ctrl)
                env.step()

        # import ipdb; ipdb.set_trace()
        repository.assign("image", image_list)
        repository.assign("state", state_list)
        repository.assign("ctrl",  ctrl_list)
        repository.close()



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()
