"""Run HMC inference over multiple scans."""
import os

os.environ[
    "XLA_FLAGS"
] = "--xla_gpu_cuda_data_dir=/opt/cuda"  # May need to change for your CUDA directory

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
from copy import deepcopy
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, loss_functions, modules
import _utils as utils

from torch.utils.data import DataLoader
from functools import partial
import wandb
import pytorch_lightning as pl
import hydra
import numpy as np
import torch

from numpyro_port import create_numpyro_model, ct_project, NUTSWithInit
import tree
from numpyro.infer import MCMC
from numpyro.infer.util import init_to_uniform
import time
from jax import numpy as jnp


@hydra.main(config_path="../conf", config_name="config")
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    assert args.uncertainty.name == "hmc"
    args.seed = (
        args.run if args.seed == 0 else int(torch.randint(0, 2**32 - 1, (1,)).item())
    )
    print(f"Setting seed to {args.seed}.")
    pl.seed_everything(args.seed)

    sidelength = args.data.sidelength

    if args.data.zero_pad:
        padding = int(((sidelength**2 + sidelength**2) ** 0.5 - sidelength) / 2) + 1
    else:
        padding = 0
    padded_sidelength = sidelength + 2 * padding

    avg_scan_uncertainty = None

    # AAPM validation set scan_ids for grid search: [0,1]
    # AAPM validation set scan_ids for fine tuning: [0,1,56]
    scan_ids = [args.data.scan_id]  # range(8)
    for scan_id in scan_ids:
        wandb.init()

        # Define data
        if args.data.name == "shepp_2d":
            img_no_pad = dataio.Shepp_2d(scan_id=scan_id)
            input_no_pad = dataio.Implicit2DWrapper(
                img_no_pad,
                sidelength=sidelength,
                preload_cuda=True,
            )
            dataloader_no_pad = DataLoader(
                input_no_pad,
                shuffle=True,
                batch_size=args.data.batch_size,
                num_workers=0,
            )
            img_data = dataio.Shepp_2d(scan_id=scan_id)
        elif args.data.name == "aapm_2d":
            img_no_pad = dataio.AAPM_2d(
                scan_id=scan_id, test_set=args.data.test_set, PADDING=0
            )
            input_no_pad = dataio.Implicit2DWrapper(
                img_no_pad,
                sidelength=sidelength,
                preload_cuda=True,
            )
            dataloader_no_pad = DataLoader(
                input_no_pad,
                shuffle=True,
                batch_size=args.data.batch_size,
                num_workers=0,
            )
            img_data = dataio.AAPM_2d(
                scan_id=scan_id,
                test_set=args.data.test_set,
                PADDING=padding,
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

        # Define the loss
        thetas = np.linspace(0.0, np.pi, args.data.n_views + 1)[:-1]
        for gt_data in dataloader:
            gt_coords = gt_data[0]["coords"]
            gt_img = gt_data[1]["img"]
            noise_sigma = args.data.noise_sigma
            if args.data.noise_snr_dB != "None":
                noise_sigma = loss_functions.get_SNR_stdev(
                    args.data.noise_snr_dB, gt_coords, gt_img, list(thetas)
                )

        # create gt_project
        gt_coords, gt_img = gt_coords.cpu().numpy(), gt_img.cpu().numpy()
        gt_project = jax.vmap(ct_project, (None, None, 0))(gt_coords[0], gt_img, thetas)

        # Define the model.
        pl.seed_everything(args.seed)
        if args.model.name == "mlp":
            model_type = args.model.model_type
            activation_type = args.model.activation_type
            omega_0 = args.model.omega_0
            net = partial(modules.UncertaINR, args.uncertainty)
            if activation_type in [
                "sine",
                "relu",
                "tanh",
                "selu",
                "elu",
                "softplus",
                "silu",
                "sigmoid",
            ]:
                if model_type in ["rbf", "nerf", "rff_enc", "mlp"]:
                    model = net(
                        type=activation_type,
                        mode=model_type,
                        omega_0=omega_0,
                        hidden_features=args.model.width,
                        out_features=args.data.out_dim,
                        num_hidden_layers=args.model.depth,
                        in_features=args.data.in_dim,
                        outermost_linear=args.model.outerlin,
                        embed_width=args.model.embed_width,
                        bias=args.model.bias,
                        zero_pad=padding,
                    )
                else:
                    raise NotImplementedError
        elif args.model.name == "grid_of_pixels":
            net = modules.GridOfPixels
            model = net(sidelength * sidelength, 1, zero_pad=padding)
        else:
            raise ValueError(f"Model type {args.model.name} not found")

        # Initialize HMC, possibly to pre-trained checkpoint.
        if args.model.name == "mlp":
            hmc_init = init_to_uniform(radius=np.sqrt(6 / args.model.width))
        elif args.model.name == "grid_of_pixels":
            hmc_init = init_to_uniform(radius=2.0)
        else:
            raise NotImplementedError
        if args.model.use_checkpoint:
            ckpts = []
            model_name = (
                args.model.name if args.model.name != "grid_of_pixels" else "gop"
            )
            for bslr_id in range(args.uncertainty.num_chains):
                if args.data.test_set:
                    ckpt_str = f"{args.model.checkpoint_dir}/{model_name}-{args.data.n_views}/scan{scan_id}_w{args.model.width}-d{args.model.depth}_bslr{bslr_id}.pth"
                else:
                    ckpt_str = f"{args.model.checkpoint_dir}/{model_name}-{args.data.n_views}/scan{scan_id}_w{args.model.width}-d{args.model.depth}_bslr{bslr_id}_val.pth"
                print(f"Using checkpoint from {ckpt_str}")
                assert os.path.exists(ckpt_str)
                ckpts.append(torch.load(ckpt_str))
        else:
            ckpts = [None]

        model_in, _ = next(iter(dataloader_no_pad))
        coords = model_in["coords"].cpu().numpy()
        model = model.cpu()
        numpyro_model = create_numpyro_model
        start = time.time()
        kernel = NUTSWithInit(numpyro_model, max_tree_depth=8, init_strategy=hmc_init)
        key = jax.random.PRNGKey(11)

        rng = np.random.default_rng(0)
        noise = []
        for _ in thetas:
            noise.append(rng.normal(scale=noise_sigma, size=(padded_sidelength, 1)))
        noise = np.stack(noise)

        ##################
        ###   RUN HMC  ###
        ##################

        for chain_id, ckpt in enumerate(ckpts):
            mcmc = MCMC(
                kernel,
                num_warmup=args.uncertainty.burn,
                num_samples=args.uncertainty.num_samples,
                num_chains=args.uncertainty.num_chains,
                progress_bar=True,
                thinning=5 if args.uncertainty.num_samples > 4 else 1,
            )
            if ckpt:
                model.load_state_dict(ckpt)
                init_params = {}
                if args.model.name == "mlp":
                    for d in range(args.model.depth + 2):
                        init_params[f"b{d}"] = model.state_dict()[
                            f"net.net.{d}.0.bias"
                        ].numpy()
                        init_params[f"w{d}"] = model.state_dict()[
                            f"net.net.{d}.0.weight"
                        ].numpy()
                elif args.model.name == "grid_of_pixels":
                    init_params["gop"] = (
                        model.state_dict()["net.0.0.weight"].reshape(-1, 1).numpy()
                    )
            else:
                init_params = None

            mcmc.run(
                key,
                coords,
                model,
                gt_project,
                gt_coords,
                thetas,
                padded_sidelength,
                noise_sigma,
                noise,
                args,
                extra_fields=(
                    "potential_energy",
                    "energy",
                    "num_steps",
                    "accept_prob",
                    "adapt_state.step_size",
                    "mean_accept_prob",
                ),
                init_params=init_params,
            )
            if chain_id == 0:
                samples = mcmc.get_samples(group_by_chain=True)
            else:
                chain_samples = mcmc.get_samples(group_by_chain=True)

                samples = tree.map_structure(
                    lambda x, y: jnp.concatenate((x, y)), samples, chain_samples
                )

        samples = tree.map_structure(lambda x: x.reshape((-1,) + x.shape[2:]), samples)
        extra_fields = mcmc.get_extra_fields()
        num_samples = len(list(samples.values())[0])
        for step in range(num_samples):
            step_fields = {k: v[step] for k, v in extra_fields.items()}
            wandb.log(step_fields)

        predictions = samples["out"]

        predictions = np.asarray(predictions)
        predictions = torch.from_numpy(predictions)
        n_samples = predictions.shape[0]
        predictions = predictions.reshape(n_samples, -1, 1)

        ##################
        #  TEST SAMPLES  #
        ##################
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)

        loss_fn = loss_functions.image_mse_uncertain
        summary_fn = partial(
            utils.write_uncertain_experiment_vals, (sidelength, sidelength)
        )
        with torch.no_grad():
            model_output = {"model_in": coords, "model_out": predictions}
            _, gt = next(iter(dataloader))
            gt["img"] = gt["img"][zero_pad_mask == 1]
            gt["img"] = torch.unsqueeze(torch.unsqueeze(gt["img"], 0), 2)
            n_samples = model_output["model_out"].shape[0]
            zero_pad_expanded = zero_pad_mask.repeat(n_samples, 1, 1)
            model_output["model_out"] = model_output["model_out"].cpu()
            model_output["model_out"] = torch.masked_select(
                model_output["model_out"], (zero_pad_expanded == 1)
            )
            model_output["model_out"] = model_output["model_out"].reshape(n_samples, -1)
            model_output["model_out"] = torch.unsqueeze(model_output["model_out"], 2)
            model_output["model_out"] = model_output["model_out"].cuda()
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
                    os.path.join(results_dir, f"{key}_{scan_id}.txt"),
                    np.array([val]),
                )

        # Modify test_loss_dct to log output for this scan_id
        keys = deepcopy(list(test_loss_dct.keys()))
        for key in keys:
            test_loss_dct[key + "_" + str(scan_id)] = test_loss_dct.pop(key)
        wandb.log(test_loss_dct)

        summary_fn(
            gt["img"],
            model_output,
            args,
            scan_id=scan_id,
            save_imgs=False,
            save_model_out=False,
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
