import gym


class CarEnvSpeedWrapper(gym.Wrapper):
    @property
    def ego_speed(self):
        # Does not exist on some envs like parking and is not useful
        try:
            return self.metrics['forward_velocity']
        except KeyError:
            return 0.
