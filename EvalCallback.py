from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import TrainFrequencyUnit
import wandb


class EvalCallback(BaseCallback):
    def __init__(self, frequency, eval_func, prefix):
        super(EvalCallback, self).__init__()
        self.frequency = frequency
        self.eval_func = eval_func
        self.prefix = prefix

    def _on_step(self) -> bool:
        step_size = 1 if not hasattr(self.training_env, 'num_envs') else self.training_env.num_envs

        op_model: OffPolicyAlgorithm = self.model

        train_freq = op_model.train_freq

        if train_freq.unit != TrainFrequencyUnit.STEP:
            raise ValueError(f"{train_freq.unit = }")

        train_freq = train_freq.frequency

        # Don't evaluate on high train frequency unless first step
        if train_freq > 1 and self.num_timesteps % train_freq != 0:
            return True

        gradient_steps = self.num_timesteps // train_freq

        if op_model.gradient_steps not in (-1, 1):
            raise ValueError(f"{op_model.gradient_steps = }")

        # Do nothing if disabled or within first step size
        if self.frequency <= 0 or gradient_steps <= step_size:
            return True

        # Break if the remainder of steps didn't wrap on last step
        if gradient_steps % self.frequency >= (gradient_steps - step_size) % self.frequency:
            return True

        assert not self.model.policy.training
        log_dict = self.eval_func(self.model)

        log_dict['step'] = self.num_timesteps
        log_dict['gradient_step'] = gradient_steps

        wandb.log({f"{self.prefix}{k}": v for k, v in log_dict.items()})
        return True
