import logging
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from trajectory_optimization.utils import upload_file_to_artifacts, wrap_env


class LapTimeCallback(BaseCallback):
    def _on_step(self) -> bool:
        info_dict = self.locals["infos"][0]  # [0] to access first env of vec env
        lap_time = info_dict.get("lap_time")
        if lap_time is not None:
            self.logger.record("lap_time", lap_time)
        return True


class VideoLogCallback(BaseCallback):
    """Upload videos written by VecVideoRecorder to wandb.

    SB3 now writes mp4s itself via moviepy, so wandb's monitor_gym hook (which patches
    gymnasium's VideoRecorder) never fires. We watch the folder and log new clips ourselves.
    A clip only lands on disk once write_videofile() has finished, so any mp4 we see is
    complete and safe to upload. No explicit step is passed so it rides wandb's internal
    counter and doesn't fight the tensorboard-synced step ordering.
    """

    def __init__(self, video_folder: str):
        super().__init__()
        self.video_folder = Path(video_folder)
        self.logged: set[Path] = set()

    def _on_step(self) -> bool:
        for video in sorted(self.video_folder.glob("*.mp4")):
            if video not in self.logged:
                self.logged.add(video)
                wandb.log({"video": wandb.Video(str(video))})
        return True


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(cfg)
    # setup wandb
    # wandb expect a primitive dict
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb_project,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
        job_type="train_agent",
    )

    def make_env():
        # instantiate a fresh env per copy so the n_envs share no state
        return wrap_env(hydra.utils.instantiate(cfg.env, _recursive_=False), cfg.wrappers)

    vec_env = DummyVecEnv([make_env for _ in range(cfg.n_envs)])

    # Record the video starting at the first step
    video_freq_steps = cfg.max_steps // cfg.video_freq
    video_folder = f"runs/{run.id}/videos"
    vec_env = VecVideoRecorder(
        vec_env,
        video_folder,
        record_video_trigger=lambda x: x % video_freq_steps == 0,
        video_length=cfg.video_length,
    )

    # Instantiate the agent
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        use_sde=cfg.use_sde,  # make sense since we're simulating a robot
        gamma=cfg.gamma,  # if we discount future rewards too much the car doesn't care about crashing in the future
        learning_starts=cfg.learning_starts,
        ent_coef=cfg.ent_coef,
        device=cfg.device,
        seed=cfg.seed,
    )

    if cfg.agent_artifact is not None:
        logging.info(f"Loading models weights from {cfg.agent_artifact}")
        artifact = wandb.use_artifact(cfg.agent_artifact)
        artifact_dir = Path(artifact.download())
        model.set_parameters(artifact_dir / "model.zip")

    try:
        # Train the agent and display a progress bar
        model.learn(
            total_timesteps=cfg.max_steps,
            progress_bar=True,
            callback=[LapTimeCallback(), VideoLogCallback(video_folder)],
        )
    except KeyboardInterrupt:
        pass

    # Save the agent
    logging.info("Saving model to artifacts")
    model_path = f"runs/{run.id}/models/model.zip"
    model.save(model_path)
    upload_file_to_artifacts(model_path, "agent", "model")

    run.finish()


if __name__ == "__main__":
    main()
