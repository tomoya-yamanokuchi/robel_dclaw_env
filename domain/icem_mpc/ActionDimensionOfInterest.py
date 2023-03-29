import numpy as np



class ActionDimensionOfInterest:
    def __init__(self,
            action_bias         : np.ndarray,
            dimension_of_interst: list,
        ):
        assert len(action_bias.shape) == 1
        self.action_bias = action_bias
        self.doi         = dimension_of_interst
        self.dim_action  = action_bias.shape[-1]
        dim_indexes      = list(np.arange(self.dim_action))
        self.dodi        = list(set(dim_indexes) - set(self.doi)) # dimension of disinterst


    def construct(self, cumsum_actions):
        assert len(cumsum_actions.shape) == 3
        num_data, step, dim_cumsum_action = cumsum_actions.shape
        assert dim_cumsum_action == len(self.doi)
        actions = self.action_bias.reshape(1, 1, -1)
        actions = np.tile(actions, (num_data, step, 1))
        actions[:, :, self.doi] += cumsum_actions
        return actions


if __name__ == '__main__':
    action_dio = ActionDimensionOfInterest(
        action_bias = np.random.randn(6,),
        dimension_of_interst = [0, 1],
    )

    actions = action_dio.construct(
        cumsum_actions = np.random.randn(300, 30, 2)
    )

    import ipdb; ipdb.set_trace()
    print(actions)
