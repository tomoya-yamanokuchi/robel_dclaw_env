@forwardable()
class RolloutBuffer(abc.Sequence):
    def_delegators("rollouts", "__len__, __iter__, __getitem__, append, extend")

    def __init__(self, *, rollouts=None, max_size=None):
        self.rollouts = _CustomList(rollouts, max_size=max_size)
        self._last_flat = None
        self._last_env_states_flat = None
        self._last_model_states_flat = None
        self.reported_flatten_warning = False

        try:
            if self.rollouts:
                _ = self.flat
        except Exception as e:
            raise TypeError(f"Concatenating rollouts failed with error {e}")

    def common_field_names(self):
        all_field_names = [r.fieldnames for r in self.rollouts]
        for f in Rollout.allowed_fields:
            if sum([1 for fields in all_field_names if f not in fields]) == 0:
                yield f

    @property
    def has_env_states(self):
        return np.sum([r.env_states is not None for r in self.rollouts]) == len(self)

    @property
    def has_model_states(self):
        return np.sum([r.model_states is not None for r in self.rollouts]) == len(self)

    @property
    def flat(self):
        if self.rollouts.modified:
            try:
                self._last_flat = np.concatenate(self.rollouts)
            except TypeError:
                # use only common fields
                common_fields = list(self.common_field_names())
                if not self.reported_flatten_warning:
                    print(f"Flatten of RolloutBuffer with different fields, use only {common_fields}")
                    self.reported_flatten_warning = True
                self._last_flat = np.concatenate([r[common_fields] for r in self.rollouts])
            if self.has_env_states:
                self._last_env_states_flat = [state for rollout in self.rollouts for state in rollout.env_states]
            if self.has_model_states:
                self._last_model_states_flat = [state for rollout in self.rollouts for state in rollout.model_states]
            self.rollouts.modified = False
        return self._last_flat

    @property
    def flat_w_states(self):
        return {'data': self.flat,
                'env_states': self._last_env_states_flat,
                'model_states': self._last_model_states_flat}

    def split(self, train_size=None, test_size=None, shuffle=True):
        train_size = train_size or 1.0 - test_size or None
        test_size = test_size or 1.0 - train_size or None

        if train_size is None and test_size is None:
            raise ValueError("At least one of train_size, test_size must be specified")

        train_rollouts, test_rollouts = train_test_split(
            self.rollouts, train_size=train_size, test_size=test_size, shuffle=shuffle
        )

        return RolloutBuffer(rollouts=train_rollouts), RolloutBuffer(rollouts=test_rollouts)

    def as_array(self, key):
        """ returns for the given field key an array of shape: rollouts, time, dim(of field) """
        try:
            if key == 'env_states' and self.has_env_states:
                return [r.env_states for r in self.rollouts]
            elif key == 'model_states' and self.has_model_states:
                return [r.model_states for r in self.rollouts]
            else:
                return np.concatenate([item[key][None, ...] for item in self], axis=0)
        except Exception as e:
            raise TypeError(
                f"Turning rollout structure into numpy array failed." f" Rollouts of unequal length? Error: {e}"
            )

    @property
    def latest_rollouts(self):
        last_rollouts = self.rollouts[-self.rollouts.number_of_latest_data_elems_added:]
        return RolloutBuffer(rollouts=last_rollouts)

    def last_n_iterations(self, num_iter=1):
        return self.last_n_rollouts(sum(self.rollouts.list_number_of_latest_data_elems_added[-num_iter:]))

    def last_n_rollouts(self, last_n=1):
        last_rollouts = self.rollouts[-last_n:]
        return RolloutBuffer(rollouts=last_rollouts)

    def n_iterations(self, start_iter=0, end_iter=1):
        return self.n_rollouts(start_n=sum(self.rollouts.list_number_of_latest_data_elems_added[:start_iter]),
                               end_n=sum(self.rollouts.list_number_of_latest_data_elems_added[:end_iter]))

    def n_rollouts(self, start_n=0, end_n=1):
        rollouts = self.rollouts[start_n:end_n]
        return RolloutBuffer(rollouts=rollouts)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == 'env_states':
                return self.flat_w_states['env_states']
            if item == 'model_states':
                return self.flat_w_states['model_states']
            return self.flat[item]
        if isinstance(item, tuple) and all(isinstance(sub_item, str) for sub_item in item):
            res = [self.__getitem__(sub_item) for sub_item in item]
            # data = tuple([self.flat[sub_item] for sub_item in item if sub_item not in ['env_states', 'model_states']])
            # if data:
            #     res.append(data)
            # if 'env_states' in item:
            #     res.append(self.flat_w_states['env_states'])
            # if 'model_states' in item:
            #     res.append(self.flat_w_states['model_states'])
            return res
        if isinstance(item, Iterable) and all(isinstance(sub_item, int) or np.isscalar(sub_item) for sub_item in item):
            return tuple([self.rollouts[sub_item] for sub_item in item])
        else:
            return self.rollouts[item]

    @property
    def mean_avg_reward(self):
        if not self.rollouts:
            return None
        return np.mean(self.flat["rewards"])

    @property
    def mean_max_reward(self):
        if not self.rollouts:
            return None
        return np.mean([np.max(rollout["rewards"]) for rollout in self.rollouts])

    @property
    def mean_return(self):
        if not self.rollouts:
            return None
        return np.mean([np.sum(rollout["rewards"]) for rollout in self.rollouts])

    @property
    def std_return(self):
        if not self.rollouts:
            return None
        if len(self.rollouts) == 1:
            return 0
        else:
            return np.std([np.sum(rollout["rewards"]) for rollout in self.rollouts])

    @property
    def is_empty(self):
        if self.rollouts._list == []:
            return True
        else:
            return False
