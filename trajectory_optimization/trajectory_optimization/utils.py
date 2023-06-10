import logging
import os
from pathlib import Path

import hydra
import wandb
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac import SAC

from trajectory_optimization.wrappers import HistoryWrapper, RescaleWrapper


def wrap_env(env, wrappers_configs):
    # if I understand correctly they should be instantiated in the same order as in the config file
    # TODO need to check if they are kept in the same order in wandb also?
    for conf in wrappers_configs.values():
        env = hydra.utils.instantiate(conf, env=env)

    env = Monitor(env)
    check_env(env)
    return env


def load_model_and_instantiate_env(artifact_alias="agent:latest", time_limit=None, **env_kwargs):
    artifact = wandb.use_artifact(artifact_alias)
    artifact_dir = Path(artifact.download())
    model = SAC.load(artifact_dir / "model.zip")

    producer_run_config = artifact.logged_by().config
    env_config = producer_run_config["env"]
    wrappers_config = producer_run_config["wrappers"]
    env = hydra.utils.instantiate(env_config, **env_kwargs, _recursive_=False)
    make_env = (
        lambda: wrap_env(env, wrappers_config)
        if time_limit is None
        else TimeLimit(wrap_env(env), max_episode_steps=time_limit)
    )

    return model, make_env()


def download_artifact_file(artifact_alias, filename):
    """Download artifact and returns path to filename.

    :param artifact_name: wandb artifact alias
    :param filename: filename in the artifact
    """
    logging.info(f"loading {filename} from {artifact_alias}")

    artifact = wandb.use_artifact(artifact_alias)
    artifact_dir = Path(artifact.download())
    filepath = artifact_dir / filename

    assert filepath.is_file(), f"{artifact_alias} doesn't contain {filename}"

    return filepath


def upload_file_to_artifacts(pth, artifact_name, artifact_type):
    logging.info(f"Saving {pth} to {artifact_name}")
    if not isinstance(pth, Path):
        pth = Path(pth)

    assert os.path.isfile(pth), f"{pth} is not a file"

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(pth)
    wandb.log_artifact(artifact)
