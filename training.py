"""Implements a generic training loop.
"""

import torch
from torch.optim.swa_utils import SWALR
import _utils as utils
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import wandb

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loss_functions


def train(
    model,
    train_dataloader,
    epochs,
    lr,
    steps_til_summary,
    epochs_til_checkpoint,
    model_dir,
    loss_fn,
    val_dataloader=None,
    double_precision=False,
    opt_cfg=None,
    save_last_ckpt=True,
    data_train_number=None,
    bslr_id=None,
    zero_pad_mask=None,
    SWA_mult=0,
):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    cosine_lr = False
    wandb.watch(model)

    # Set up optimizer.
    if opt_cfg is not None:
        if opt_cfg.name == "sgd":
            optim = torch.optim.SGD(
                lr=lr,
                params=model.parameters(),
                weight_decay=opt_cfg.weight_decay,
                momentum=opt_cfg.momentum,
            )
        elif opt_cfg.name == "adam":
            optim = torch.optim.Adam(
                lr=lr, params=model.parameters(), weight_decay=opt_cfg.weight_decay
            )
        elif opt_cfg.name == "adamw":
            optim = torch.optim.AdamW(
                lr=lr, params=model.parameters(), weight_decay=opt_cfg.weight_decay
            )

        if cosine_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        if SWA_mult > 0:
            swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_start = int(0.75 * epochs)
            swa_scheduler = SWALR(optim, swa_lr=lr * SWA_mult)

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, "summaries")
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    results_dir = os.path.join(model_dir, "results")
    utils.cond_mkdir(results_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        val_losses = []

        epoch = 0
        while epoch < epochs:
            if SWA_mult > 0:
                if epoch > swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                elif cosine_lr:
                    scheduler.step()
            elif cosine_lr:
                scheduler.step()
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoints_dir, "model_epoch_%04d.pth" % epoch),
                )
                np.savetxt(
                    os.path.join(results_dir, "train_losses_epoch_%04d.txt" % epoch),
                    np.array(train_losses),
                )

            for _, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {
                        key: value.double() for key, value in model_input.items()
                    }
                    gt = {key: value.double() for key, value in gt.items()}

                optim.zero_grad()
                model_output = model(model_input)
                losses = loss_fn(model_output)

                train_loss = torch.tensor(0.0).cuda()
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    train_loss += single_loss
                    wandb.log(
                        {"step": total_steps, loss_name: single_loss.item()},
                        commit=False,
                    )

                train_losses.append(train_loss.item())

                model_out_temp = model_output["model_out"].clone()
                model_output["model_out"] = model_output["model_out"][
                    zero_pad_mask == 1
                ]
                model_output["model_out"] = torch.unsqueeze(
                    torch.unsqueeze(model_output["model_out"], 0), 2
                )

                train_loss.backward()

                optim.step()

                # Log useful metrics and quantities every iteration
                with torch.no_grad():
                    img_loss = loss_functions.image_mse(None, model_output, gt)[
                        "img_loss"
                    ]
                    img_snr = loss_functions.get_SNR(
                        gt["img"], model_output["model_out"]
                    )

                    if data_train_number is None:
                        wandb.log({"img_loss": img_loss})
                        wandb.log({"img_psnr": -10 * torch.log10(img_loss)})
                        wandb.log({"img_snr": img_snr})
                    else:
                        wandb.log({"img_loss_" + str(data_train_number): img_loss})
                        wandb.log(
                            {
                                "img_psnr_"
                                + str(data_train_number): -10 * torch.log10(img_loss)
                            }
                        )
                        wandb.log({"img_snr_" + str(data_train_number): img_snr})

                pbar.update(1)

                model_output["model_out"] = model_out_temp

                # Track progress and log test metrics periodically.
                if not total_steps % steps_til_summary:
                    tqdm.write(
                        "Epoch %d, Total loss %0.6f, iteration time %0.6f"
                        % (epoch, train_loss, time.time() - start_time)
                    )

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_loss = 0
                            for (model_input, gt) in val_dataloader:
                                model_input = {
                                    key: value.cuda()
                                    for key, value in model_input.items()
                                }
                                gt = {key: value.cuda() for key, value in gt.items()}
                                model_output = model(model_input)
                                val_loss_dct = loss_fn(model_output)  # , gt)
                                for _, v in val_loss_dct.items():
                                    val_loss += v.item()
                            val_losses.append(val_loss)

                            wandb.log(
                                {"step": total_steps, "val_loss": val_loss},
                                commit=False,
                            )

                        model.train()
                total_steps += 1
            epoch += 1

        if SWA_mult > 0:
            torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
        if save_last_ckpt:
            if data_train_number is not None or bslr_id is not None:
                text_temp = ""
                if data_train_number is not None:
                    text_temp += "_" + str(data_train_number)
                if bslr_id is not None:
                    text_temp += "_" + str(bslr_id)
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoints_dir, "model_final" + text_temp + ".pth"),
                )
            else:
                torch.save(
                    model.state_dict(), os.path.join(checkpoints_dir, "model_final.pth")
                )
        return model
