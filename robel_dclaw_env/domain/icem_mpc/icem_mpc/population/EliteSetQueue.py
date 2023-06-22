from collections import deque


class EliteSetQueue:
    def __init__(self, config):
        self.num_elite           = config.num_elite
        self.fraction_rate_elite = config.fraction_rate_elite
        self.num_reuse           = int(self.num_elite * self.fraction_rate_elite)
        self.elites              = deque([])

    def append(self, elites):
        assert elites.shape[0] == self.num_elite
        self.elites.append(elites)

    def get_elites(self):
        return self.elites.popleft()[:self.num_reuse]

    def get_shifted_elites(self):
        return self.elites.popleft()[:, 1:][:self.num_reuse]

    def is_empty(self):
        return len(self.elites) == 0



if __name__ == '__main__':
    import numpy as np
    elite = EliteSetQueue(10, 0.3)

    print(elite.is_empty())

    elite.append(np.random.randn(10, 30, 2))
    print(elite.is_empty())
    print(elite.get_elites().shape)

    elite.append(np.random.randn(10, 30, 2))
    print(elite.is_empty())
    print(elite.get_shifted_elites().shape)

    print(elite.is_empty())
