import os
import sys

import numpy as np
import wandb

import torch
from Architecture import SetFeatureExtractor, FlatFeatureExtractor
from CarEnv import CarEnv


def make_env_car(**kwargs):
    from Util import TorchEnvObsWrapper
    from Wrappers.CarEnvSpeedWrapper import CarEnvSpeedWrapper
    result = TorchEnvObsWrapper(CarEnvSpeedWrapper(CarEnv(**kwargs)))

    return result


def make_env_highway(**kwargs):
    from HighwayCompat import HighwayCompatEnv, HighwayCompatWrapper

    kwargs["offscreen_rendering"] = True
    result = HighwayCompatWrapper(HighwayCompatEnv(**kwargs))

    return result


def evaluate(env, agent, episodes=100):
    from tqdm import trange
    rewards = []
    etrs = []
    frames = []
    velocities = []

    for episode in trange(episodes, desc="Evaluating"):
        current_reward = 0.
        current_velocities = []

        # Reseed with episode number before each episode to have as deterministic environment sequence as possible
        env.seed(episode)
        obs = env.reset()
        early_termination = False

        while True:
            act, _ = agent.predict(obs, deterministic=True)

            if episode == 0:
                frames.append(env.render(mode="rgb_array"))

            obs, reward, done, info = env.step(act)
            truncated = info.get('TimeLimit.truncated', False)

            current_reward += reward

            if done:
                if not truncated and reward < -.5:
                    early_termination = True
                break
            else:
                current_velocities.append(env.ego_speed)

        etrs.append(early_termination)
        rewards.append(current_reward)
        if len(current_velocities) > 0:
            velocities.append(np.mean(current_velocities))

    return {
        "avg_reward": np.mean(rewards),
        "avg_etr": np.mean(etrs),
        "avg_velocity": np.mean(velocities) if len(velocities) > 0 else 0.,
        "video": wandb.Video(np.transpose(np.stack(frames), (0, 3, 1, 2)), fps=int(1 / env.dt))
    }


def main():
    import wandb
    from argparse import ArgumentParser
    from stable_baselines3.common.utils import configure
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import ProgressBarCallback
    from EvalCallback import EvalCallback
    from functools import partial
    from importlib import import_module
    from time import time
    # Fail fast if we cannot import moviepy
    import_module("moviepy.editor")

    parser = ArgumentParser()
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--eval_frequency', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=.95)
    parser.add_argument('--problem', default='parking')
    parser.add_argument('--extractor', default='deepset')
    args = parser.parse_args()

    print(f"{torch.cuda.is_available() = }", file=sys.stderr)

    env_args = {}

    train_eval_eps = 100
    cnn_conf = None
    cnn_resolution = None

    if args.problem == "parking":
        from CarEnv.Configs import PARALLEL_PARKING
        env_args['config'] = PARALLEL_PARKING
        make_env = make_env_car
    elif args.problem == "racing_fast":
        from CarEnv.Configs import RACING_FAST
        env_args['config'] = RACING_FAST
        make_env = make_env_car
    elif args.problem == "highway":
        train_eval_eps = 200
        env_args = {}
        make_env = make_env_highway
    else:
        raise ValueError(f"{args.problem = }")

    log_id = str(int(time()))
    log_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp', log_id)
    os.makedirs(log_root, exist_ok=False)
    wandb.init(project="LffDs", resume='never', dir=log_root, id=log_id, config=args, sync_tensorboard=True)

    path = f"tmp/{args.problem}.torch"
    path = os.path.join(wandb.run.dir, path)
    env_path = path + ".env"

    preencoder = None

    if args.extractor == 'deepset':
        preencoder = 'fc'
    elif args.extractor == 'lffds':
        if args.problem == 'racing_fast' or args.problem == 'parking':
            preencoder = 'ffn_lin_30'
        else:
            preencoder = 'ffn_lin_3'

    if args.extractor == 'deepset' or args.extractor == 'lffds':
        policy_kwargs = {
            "features_extractor_class": SetFeatureExtractor,
            "features_extractor_kwargs": {
                "preencoder": preencoder,
            }
        }

        # The default legacy architecture for this problem
        if args.problem == 'racing_fast' or "parking" in args.problem:
            policy_kwargs["features_extractor_kwargs"]["item_arch"] = {
                "cones_set": [64, 32, 128],
            }
        elif args.problem == 'highway':
            policy_kwargs["features_extractor_kwargs"]["item_arch"] = {
                "other_set": [128, 64, 256],
            }
            policy_kwargs["features_extractor_kwargs"]["static_dims"] = 64
    elif args.extractor == 'flat':
        policy_kwargs = {
            "features_extractor_class": FlatFeatureExtractor,
            "features_extractor_kwargs": {}
        }

        if args.problem == 'racing_fast' or "parking" in args.problem:
            policy_kwargs["features_extractor_kwargs"]["set_arch"] = {"cones_set": [512, 256]}
        elif args.problem == 'highway':
            policy_kwargs["features_extractor_kwargs"]["set_arch"] = {"other_set": [256, 256]}
            policy_kwargs["features_extractor_kwargs"]["static_dims"] = 64
    elif args.extractor == 'cnn':
        from Architecture import CnnFeaturesExtractor
        from Wrappers.RacingCnnWrapper import RacingCnnWrapper

        if cnn_conf is None:
            if args.problem == "highway":
                # Determined to be the best config on highway
                cnn_resolution = 3.
                cnn_conf = ((32, (1, 7), (1, 3)), (16, (1, 7), (1, 3)), (16, (2, 3), (1, 2)))
                print(f"Using {cnn_resolution = }, {cnn_conf = }", file=sys.stderr)

        policy_kwargs = {
            "features_extractor_class": CnnFeaturesExtractor,
            "features_extractor_kwargs": {
                "conf": cnn_conf,
            },
        }

        if args.problem.startswith('racing') or 'parking' in args.problem:
            old_make_env = make_env

            def _wrap_env(*args, **kwargs):
                new_env = old_make_env(*args, **kwargs)
                return RacingCnnWrapper(new_env)

            make_env = _wrap_env
        elif args.problem == "highway":
            from HighwayCompat import HighwayCnnWrapper
            old_make_env = make_env

            def _wrap_env(*args, **kwargs):
                new_env = old_make_env(*args, **kwargs)
                return HighwayCnnWrapper(new_env, resolution=cnn_resolution)

            make_env = _wrap_env
    else:
        raise ValueError(f"{args.extractor = }")

    env = make_env(**env_args)
    eval_env = make_env(**env_args)

    obs_shapes = {k: v.shape for k, v in eval_env.observation_space.spaces.items()}
    print(f"{obs_shapes = }", file=sys.stderr)

    callbacks = [
        EvalCallback(args.eval_frequency, partial(evaluate, eval_env, episodes=train_eval_eps), "train_eval/"),
        ProgressBarCallback(),
    ]

    from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

    act_shape = env.action_space.shape
    noise = OrnsteinUhlenbeckActionNoise(np.zeros(act_shape), np.ones(act_shape) * .2, dt=1e-1)

    # No OU noise on highway or lidar
    if args.problem == "highway" or "parking" in args.problem:
        noise = None

    print(f"Using noise: {noise = }", file=sys.stderr)

    agent = SAC("MultiInputPolicy", env, action_noise=noise, gamma=args.gamma, buffer_size=1_000_000,
                policy_kwargs=policy_kwargs, verbose=2)
    # Must configure logging exactly (!) like this or wandb fails to sync if moved to any other folder
    agent.set_logger(configure(wandb.run.dir, ["tensorboard"]))

    target_steps = args.steps * 1
    agent.learn(target_steps, callback=callbacks)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(env_args, env_path)
    agent.save(path)

    eval_episodes = 100

    # Very noisy problem
    if args.problem == "highway":
        eval_episodes = 500

    agent.policy.set_training_mode(False)
    log_dict = evaluate(eval_env, agent, episodes=eval_episodes)

    wandb.log({f"eval/{k}": v for k, v in log_dict.items()})
    wandb.finish()


if __name__ == '__main__':
    main()
