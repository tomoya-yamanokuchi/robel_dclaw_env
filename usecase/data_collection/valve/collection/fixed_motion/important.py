        config.env.camera.z_distance = 0.4 # ロボットがフレームアウトする情報欠損を起こさないようにカメラを引きで設定
        env_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        task_space_position_init = np.array(
            [
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
            ]
        )
        num_action_variation = task_space_position_init.shape[0]
        init_state   = EnvState(
            robot_position        = np.tile(np.array(config.env.robot_position_init).reshape(1, -1), (num_action_variation, 1)),
            robot_velocity        = np.tile(np.array(config.env.robot_velocity_init).reshape(1, -1), (num_action_variation, 1)),
            object_position       = np.tile(np.array(config.env.object_position_init).reshape(1, -1), (num_action_variation, 1)),
            object_velocity       = np.tile(np.array(config.env.object_velocity_init).reshape(1, -1), (num_action_variation, 1)),
            end_effector_position = None,
            task_space_positioin  = task_space_position_init,
            mode  = "sequence",
        )
        # import ipdb; ipdb.set_trace()

        step = config.run.step
        dim_task_space = 3

        # -----------------------------------------------------------------------------
        #                                    ctrl
        # -----------------------------------------------------------------------------
        const = 0.05
        # import ipdb; ipdb.set_trace()
        ctrl_task_diff = np.stack(
            (
                np.stack((np.zeros(step) + const, np.zeros(step),           np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step) - const, np.zeros(step),           np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step) + const,   np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step) - const,   np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step),           np.zeros(step) + const) , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step),           np.zeros(step) - const) , axis=-1),
            ), axis=0
        )
        num_ctrl = ctrl_task_diff.shape[0]

        # init_state =
        # import ipdb; ipdb.set_trace()

        chunked_input = []
        for i in range(config.run.sequence):
            chunked_input_unit_dict = {
                "ctrl_task_diff" : ctrl_task_diff,
                "init_state"     : init_state,
            }
            chunked_input.append(chunked_input_unit_dict)
