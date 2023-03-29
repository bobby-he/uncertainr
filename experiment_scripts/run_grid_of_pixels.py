# Enable import from parent package
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, training, loss_functions, modules
import _utils as utils

from torch.utils.data import DataLoader
from functools import partial
import wandb
import pytorch_lightning as pl
import hydra
import numpy as np
import torch
from copy import deepcopy


@hydra.main(config_path="../conf", config_name="config")
def main(args):
    assert args.model.name == "grid_of_pixels"
    root_path = os.getcwd()
    args.seed = (
        args.run if args.seed == 0 else int(torch.randint(0, 2**32 - 1, (1,)).item())
    )
    print(f"Setting seed to {args.seed}.")
    pl.seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sidelength = args.data.sidelength
    padding = int(((sidelength**2 + sidelength**2) ** 0.5 - sidelength) / 2) + 1
    padded_sidelength = sidelength + 2 * padding

    # for averaging the uncertainty across all scan_ids
    avg_scan_uncertainty = None

    scan_ids = [args.data.scan_id]  # range(0, 8)
    for scan_id in scan_ids:
        wandb.init()
        if args.data.name == "shepp_2d":
            img_data = dataio.Shepp_2d(scan_id=scan_id)
            img_no_pad = img_data
        elif args.data.name == "aapm_2d":
            img_data = dataio.AAPM_2d(
                scan_id=scan_id,
                test_set=args.data.test_set,
                PADDING=padding,
            )
            img_no_pad = dataio.AAPM_2d(
                scan_id=scan_id, test_set=args.data.test_set, PADDING=0
            )
        else:
            raise ValueError(f"Dataset {args.data.name} not implemented.")
        coord_data = dataio.Implicit2DWrapper(
            img_data, sidelength=padded_sidelength, preload_cuda=True
        )
        zero_pad_mask = dataio.zero_pad_mask(padding, padded_sidelength, coord_data)
        zero_pad_mask = torch.unsqueeze(torch.unsqueeze(zero_pad_mask, 0), 2)
        zero_pad_mask = zero_pad_mask.cuda()
        input_data = dataio.Implicit2DWrapper(
            img_data, sidelength=padded_sidelength, preload_cuda=True
        )
        dataloader = DataLoader(
            input_data, shuffle=True, batch_size=args.data.batch_size, num_workers=0
        )
        input_no_pad = dataio.Implicit2DWrapper(
            img_no_pad,
            sidelength=sidelength,
            preload_cuda=True,
        )
        dataloader_no_pad = DataLoader(
            input_no_pad, shuffle=True, batch_size=args.data.batch_size, num_workers=0
        )

        ###################
        ### TRAIN MODEL ###
        ###################
        net = modules.GridOfPixels
        model = net(sidelength * sidelength, 1, zero_pad=padding)

        model.cuda()

        # Define the loss
        thetas = np.linspace(0.0, np.pi, args.data.n_views + 1)[:-1]

        for data in dataloader:
            gt_coords = data[0]["coords"]
            gt_img = data[1]["img"]

            noise_sigma = args.data.noise_sigma
            if args.data.noise_snr_dB != "None":
                noise_sigma = loss_functions.get_SNR_stdev(
                    args.data.noise_snr_dB, gt_coords, gt_img, list(thetas)
                )
            loss_fn = loss_functions.single_image_mse_project(
                gt_coords,
                gt_img,
                list(thetas),
                args.data.view_batch_size,
                projection_length=padded_sidelength,
                noise_sigma=noise_sigma,
                regularize=args.reg.type,
                reg_coeff=args.reg.coeff,
            )

        training.train(
            model=model,
            train_dataloader=dataloader_no_pad,
            epochs=args.num_epochs,
            lr=args.opt.lr,
            steps_til_summary=args.steps_til_summary,
            epochs_til_checkpoint=args.epochs_til_ckpt,
            model_dir=root_path,
            loss_fn=loss_fn,
            opt_cfg=args.opt,
            SWA_mult=args.model.swa_mult,
            zero_pad_mask=zero_pad_mask,
        )

        ##################
        ### TEST MODEL ###
        ##################
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        loss_fn = loss_functions.image_mse_uncertain
        summary_fn = partial(
            utils.write_uncertain_experiment_vals, (sidelength, sidelength)
        )
        with torch.no_grad():
            model_input, gt = next(iter(dataloader))
            model_output = model(model_input)
            gt["img"] = gt["img"][zero_pad_mask == 1]
            gt["img"] = torch.unsqueeze(torch.unsqueeze(gt["img"], 0), 2)
            model_output["model_out"] = model_output["model_out"][zero_pad_mask == 1]
            model_output["model_out"] = torch.unsqueeze(
                torch.unsqueeze(model_output["model_out"], 0), 2
            )

            model_output["var_out"] = model_output["model_out"].var(0, keepdim=True)
            model_output["mean_out"] = model_output["model_out"].mean(0, keepdim=True)
            test_loss_dct = loss_fn(
                model_output,
                gt,
                coverage_mse=True,
                C_save_path=results_dir + "/",
                scan_id=scan_id,
            )

            if avg_scan_uncertainty is None:
                avg_scan_uncertainty = deepcopy(test_loss_dct)
            else:
                for key, val in test_loss_dct.items():
                    avg_scan_uncertainty[key] += val
            for key, val in test_loss_dct.items():
                np.savetxt(
                    os.path.join(results_dir, f"{key}_{scan_id}.txt"), np.array([val])
                )

        # Modify test_loss_dct to log output for this scan_id
        keys = deepcopy(list(test_loss_dct.keys()))
        for key in keys:
            test_loss_dct[key + "_" + str(scan_id)] = test_loss_dct.pop(key)
        wandb.log(test_loss_dct)

        # Finish with saving images
        summary_fn(
            gt["img"],
            model_output,
            args,
            scan_id=scan_id,
            save_imgs=False,
            save_model_out=True,
        )

        torch.cuda.empty_cache()

    for key, val in avg_scan_uncertainty.items():
        avg_scan_uncertainty[key] = val / len(scan_ids)
        np.savetxt(
            os.path.join(results_dir, f"{key}_avg.txt"),
            np.array([avg_scan_uncertainty[key]]),
        )
    wandb.log(avg_scan_uncertainty)
    wandb.close()


if __name__ == "__main__":
    main()
