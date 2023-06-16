import logging

import hydra
import numpy as np
import torch
import torchvision
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
from tqdm import tqdm

from trajectory_prediction.utils import (
    load_weights_from_artifacts,
    save_model_to_artifacts,
    seed,
)
from trajectory_prediction.viz import make_vid_from_2d_traj, plot_traj


def instantiate_transforms(confs):
    augs = []
    for conf in confs.values():
        # augs expect primitive types so we use _convert_
        augs.append(hydra.utils.instantiate(conf, _convert_="all"))

    return torchvision.transforms.Compose(augs)


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(cfg)

    # setup wandb
    # wandb expect a primitive dict
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project, save_code=True, job_type=cfg.wandb.job_type, config=config
    )

    accelerator = Accelerator()

    seed(cfg.seed, cfg.cudnn_deterministic)

    # setup model
    model = hydra.utils.instantiate(cfg.model)

    # if artifact specified, load weights from artifact
    if cfg.model_artifact_name:
        logging.info(f"Loading weights from artifact: {cfg.model_artifact_name}")
        load_weights_from_artifacts(model, cfg.model_artifact_name)

    base_transform = instantiate_transforms(cfg.augmentations.base_transforms)
    transform = instantiate_transforms(cfg.augmentations.augs_transforms)

    # setup the dataset
    raw_ds = hydra.utils.instantiate(cfg.dataset.train, transform=transform)
    test_ds = hydra.utils.instantiate(cfg.dataset.test, transform=base_transform)

    # sim seq are 100 len so 300 should give us the first 3 seq
    # since nothing is shuffled yet images should be in the right order
    viz_ds, ds = torch.utils.data.Subset(raw_ds, range(0, 300)), torch.utils.data.Subset(
        raw_ds, range(300, len(raw_ds))
    )

    # randomly split ds into train_ds and val_ds
    val_size = int(len(ds) * cfg.val_pct)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - val_size, val_size])

    # TODO make this a lil bit less ugly?
    # ensure val_ds doesn't have augs
    val_ds.transform = base_transform

    if cfg.overfit_batches is not None:
        # only keep the first n batches
        train_ds = torch.utils.data.Subset(train_ds, range(cfg.overfit_batches * cfg.batch_size))

    # setup dataloaders
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    viz_dl = torch.utils.data.DataLoader(
        viz_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # setup optimizer
    # TODO parametrize?
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # setup loss
    # TODO parametrize
    criterion = hydra.utils.instantiate(cfg.loss)

    train_dl, val_dl, viz_dl, test_dl, model, optimizer = accelerator.prepare(
        train_dl, val_dl, viz_dl, test_dl, model, optimizer
    )
    model.to(accelerator.device)

    # defined in context to avoid passing everything
    def evaluate(dl, step_name, sanity_check=False):
        """step to know where to log."""
        losses = []

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(dl):
                # forward
                loss, y, y_hat = step(batch, model, criterion)
                losses.append(loss)

                # at least two batches
                if sanity_check and idx > 0:
                    break

        mean_loss = torch.tensor(losses).mean()
        model.train()

        if not sanity_check:
            wandb.log(
                {
                    f"{step_name}/loss": mean_loss,
                }
            )

    # run validation and test before training
    # to make sure everything is working
    logging.info("Sanity check on val")
    evaluate(val_dl, "test", sanity_check=True)
    logging.info("Sanity check viz on test")
    viz(test_dl, model, key="test")

    # train
    try:
        for epoch in range(cfg.epochs):
            logging.info(f"Epoch {epoch}")

            for batch_idx, batch in enumerate(tqdm(train_dl)):
                loss, _, _ = step(batch, model, criterion)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                wandb.log({"train/loss": loss})

                if (batch_idx != 0 or epoch != 0) and cfg.overfit_batches is None:
                    if batch_idx % (len(train_dl) // cfg.val_freq) == 0:
                        logging.info("Validating")
                        evaluate(val_dl, "val")
                        viz(viz_dl, model)
                        viz(test_dl, model, key="test")

    except KeyboardInterrupt:
        logging.info("Caught KeyboardInterrupt")

    logging.info("Saving model...")
    model = (
        model.cpu()
    )  # get it off mps else earlier torch don't recognize it and we can't load on jetson
    save_model_to_artifacts(model, "model")

    run.finish()


def step(batch, model, criterion):
    # get data
    x, y = batch["image"], batch["trajectory"]
    speed = batch["speed"]

    # forward
    y_hat = model(x, speed)
    loss = criterion(y_hat, y)

    return loss, y, y_hat


def viz(dl, model, key="val"):
    model.eval()
    images = []
    gt_traj = []
    pred_traj = []
    with torch.no_grad():
        for idx, batch in enumerate(dl):
            # forward
            x = batch["image"]

            if key != "test":
                y = batch["trajectory"]
                gt_traj.append(y.cpu().numpy())

                speed = batch["speed"]
            else:
                # dummy speed for test ds bc we didn't collect speed
                # TODO would be nice to record the real speed
                # speed is between -1, 1 so lets just use zero
                speed = torch.zeros((x.size(0), 1))
                speed = speed.to(x.device)

            y_hat = model(x, speed)

            # TODO this is ugly can we fix it?
            pred_traj.append(y_hat.cpu().numpy())
            images.append(x.cpu().numpy())

    model.train()

    # make the viz
    images = np.concatenate(images, axis=0)
    images = np.ascontiguousarray(images.transpose(0, 2, 3, 1))  # cv2 images are WHC

    # denormalize images
    images_min, images_max = images.min(), images.max()
    images = ((images - images_min) / (images_max - images_min) * 255).astype(np.uint8)

    if key != "test":
        gt_traj = np.concatenate(gt_traj, axis=0).reshape(
            images.shape[0], -1, 4
        )  # (images, nb points traj, xy)
    pred_traj = np.concatenate(pred_traj, axis=0).reshape(images.shape[0], -1, 4)

    for idx in range(images.shape[0]):
        if key != "test":  # only plot gt when we have it
            plot_traj(images[idx], gt_traj[idx, :, 0:2], color=(0, 255, 0))
        plot_traj(images[idx], pred_traj[idx, :, 0:2], color=(0, 0, 255))

    if key != "test":  # plot gt and pred
        vid_2d = make_vid_from_2d_traj(pred_traj, gt_traj)
    else:
        vid_2d = make_vid_from_2d_traj(pred_traj)

    wandb.log(
        {
            f"{key}/viz": wandb.Video(
                images.transpose(0, 3, 1, 2), fps=10
            ),  # actual fps is 40 but we want to viz more slowly TODO should be configured somewhere
            f"{key}/2d": wandb.Video(vid_2d.transpose(0, 3, 1, 2), fps=10),
        }
    )


if __name__ == "__main__":
    main()
