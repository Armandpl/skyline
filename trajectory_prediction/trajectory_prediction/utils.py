import os
import random

import numpy as np
import torch
import wandb


class DictAccumulator:
    """passing a dict of metrics to accumulate if the value doesn't exists yet set it to zero
    calling compute resets values."""

    def __init__(self):
        self._reset()

    def __call__(self, dict):
        self.count += 1

        for k, v in dict.items():
            # add v to k if k exists
            # else k = 0
            self.state[k] = self.state.get(k, 0) + v

    def _reset(self):
        self.state = {}
        self.count = 0

    def compute(self):
        # compute mean
        for k, v in self.state.items():
            self.state[k] = v / self.count

        # return a copy of state
        state = self.state.copy()

        self._reset()
        return state


def seed(seed, cudnn_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def save_model_to_artifacts(model, artifact_name="model"):
    model_path = os.path.join(wandb.run.dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(model_path)

    wandb.log_artifact(artifact)


def load_weights_from_artifacts(model, artifact_name):
    artifact = wandb.use_artifact(artifact_name)
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, "model.pt")
    model.load_state_dict(torch.load(model_path))
