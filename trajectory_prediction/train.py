import logging

import hydra
import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch
import torchvision
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from trajectory_prediction.utils import (
    DictAccumulator,
    load_weights_from_artifacts,
    save_model_to_artifacts,
    seed,
)


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(cfg)

    # setup wandb
    # wandb expect a primitive dict
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(
        project=cfg.wandb.project, save_code=True, job_type=cfg.wandb.job_type, config=config
    )

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=cfg.gradient_accumulation_steps, log_with="wandb"
    # )

    seed(cfg.seed, cfg.cudnn_deterministic)

    # setup model
    model = hydra.utils.instantiate(cfg.model)
    transform = model.get_transforms()

    # if artifact specified, load weights from artifact
    if cfg.model_artifact_name:
        logging.info(f"Loading weights from artifact: {cfg.model_artifact_name}")
        load_weights_from_artifacts(model, cfg.model_artifact_name)

    # setup the dataset
    ds = hydra.utils.instantiate(cfg.dataset.train, transform=transform)

    # randomly split ds into train_ds and val_ds
    val_size = int(len(ds) * cfg.val_pct)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - val_size, val_size])

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

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # setup loss
    criterion = torch.nn.MSELoss()
    train_metrics = DictAccumulator()

    #train_dl, val_dl, model, optimizer = accelerator.prepare(train_dl, val_dl, model, optimizer)
    model.to("cuda:0")

    def evaluate(dl, step_name, sanity_check=False):
        """step to know where to log."""
        losses = []

        model.eval()
        with torch.no_grad():
            for idx, (images, targets) in enumerate(dl):
                # forward
                loss, y, y_hat = step((images, targets), model, criterion)
                losses.append(loss)

                # at least two batches
                if sanity_check and idx > 0:
                    break
                
                for image, angle_prediction, angle_gt in zip(images, y_hat, y):
                    # Convert the image tensor to a NumPy array
                    image_array = image.numpy().transpose((1, 2, 0))
                    
                    #denormalize compared to imagenet
                    image_array = image_array * np.array([0.229, 0.224, 0.225]) + np.array(
                        [0.485, 0.456, 0.406]
                    )

                    # Convert the NumPy array to a PIL Image
                    pil_image = PIL.Image.fromarray((image_array * 255).astype(np.uint8))

                    # Draw the predicted point on the image
                    draw = PIL.ImageDraw.Draw(pil_image)
                    x = (angle_prediction + 1)/2 * 224  # Scale from -1.0-1.0 to image size
                    x = 224 - x  # Invert x-axis
                    y = 112  # Center of the y-axis
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")

                    x = (angle_gt + 1)/2 * 224  # Scale from -1.0-1.0 to image size
                    x = 224 - x  # Invert x-axis
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="blue")

                    # Log the image with the prediction to W&B
                    logged_image = wandb.Image(pil_image, caption=f"Prediction: {angle_prediction}")
                    wandb.log({f"{step_name}/image": logged_image})
                    break

        mean_loss = torch.tensor(losses).mean()
        model.train()

        if not sanity_check:
            wandb.log(
                {
                    f"{step_name}/loss": mean_loss,
                    # f"{step_name}/percent_err_vs_all_zeros": percent_err_vs_all_zeros,
                }
            )


    # run validation and test before training
    # to make sure everything is working
    logging.info("Sanity check on val")
    evaluate(val_dl, "test", sanity_check=True)

    # train
    try:
        for epoch in range(cfg.epochs):
            logging.info(f"Epoch {epoch}")

            for batch_idx, batch in enumerate(tqdm(train_dl)):
                #with accelerator.accumulate(model):
                loss, _, _ = step(batch, model, criterion, train_metrics)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if batch_idx != 0 or epoch != 0:
                    if batch_idx % cfg.gradient_accumulation_steps == 0:
                        wandb.log({"train/loss": loss})

                if cfg.overfit_batches is None:
                    if batch_idx % (len(train_dl) // cfg.val_freq) == 0:
                        logging.info("Validating")
                        evaluate(val_dl, "val")

    except KeyboardInterrupt:
        logging.info("Caught KeyboardInterrupt")

    logging.info("Saving model...")
    save_model_to_artifacts(model, "model")

    run.finish()


def step(batch, model, criterion, metrics=None):
    # get data
    x, y = batch
    x = x.to("cuda:0")
    y = y.to("cuda:0")

    # forward
    y_hat = model(x)
    y_hat = torch.tanh(y_hat)
    loss = criterion(y_hat, y)

    if metrics is not None:
        losses = {
            "loss": loss,
        }
        metrics(losses)

    return loss, y, y_hat


# TODO take real world image and plot steering on them

if __name__ == "__main__":
    main()
