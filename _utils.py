import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import loss_functions
import os
from scipy.stats import norm
import pickle


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_uncertain_experiment_vals(
    image_resolution,
    gt,
    model_output,
    args,
    suffix="",
    scan_id=None,
    save_imgs=True,
    save_model_out=True,
    save_path=None,
):
    if save_path is None:
        img_path = "plots/" + str(scan_id) + "/"
        model_out_path = "model_out/" + str(scan_id) + "/"
    else:
        if scan_id is not None:
            img_path = save_path + "plots/" + str(scan_id) + "/"
            model_out_path = save_path + "model_out/" + str(scan_id) + "/"
        else:
            img_path = save_path + "plots/"
            model_out_path = save_path + "model_out/"

    vars = model_output["var_out"] + loss_functions.EPS
    mean_preds = model_output["mean_out"]

    if save_imgs:
        os.makedirs(img_path, exist_ok=True)
    if save_model_out:
        os.makedirs(model_out_path, exist_ok=True)

    """Make various summary plots from a CT run"""
    # Calculate pixel-wise metrics
    sqdiffs = (mean_preds - gt) ** 2
    nlls = (sqdiffs / vars + torch.log(2 * np.pi * vars)) / 2

    gt_img = (
        dataio.lin2img(gt, image_resolution)
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    pred_img = (
        dataio.lin2img(mean_preds, image_resolution)
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    var_img = (
        dataio.lin2img(vars, image_resolution)
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    sqdiff_img = (
        dataio.lin2img(sqdiffs, image_resolution)
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    nll_img = (
        dataio.lin2img(nlls, image_resolution)
        .permute(0, 2, 3, 1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    coverage_img = norm.cdf(np.sqrt(sqdiff_img / var_img))
    sym_coverage_img = norm.cdf(np.sqrt(sqdiff_img / var_img)) - norm.cdf(
        -np.sqrt(sqdiff_img / var_img)
    )

    plot_list = [
        (0, 1, "mean_preds", "Predicted mean", pred_img),
        (-5, 5, "nll", "NLL", nll_img),
        (0, 0.05, "squared_err", "Squared error vs Ground Truth", sqdiff_img),
        (0, 0.05, "var_preds", "Predicted variance", var_img),
        (0, 1, "ground_truth", "Ground Truth", gt_img),
        (0, 1, "coverage", "Coverage probability", coverage_img),
        (0, 1, "sym_coverage", "Symmetric coverage prob", sym_coverage_img),
    ]

    model_str = f"Model={args.model.model_type}"
    uncertainty_str = f"Uncertainty={args.uncertainty.name}"
    scan_str = f"Scan={args.data.scan_id}"
    if "n_views" in args.data:
        view_str = f"#Views={args.data.n_views}"
    elif "n_samp" in args.data:
        view_str = f"#Samples={args.data.n_samp}"

    if save_imgs:
        # Plot pixel-wise images of metrics
        for lower, upper, name, nice_name, img in plot_list:
            plt.imshow(img[:, :, 0])
            plt.title(nice_name, fontsize=20)
            plt.ylabel(model_str + ", " + scan_str, fontsize=15)
            plt.xlabel(view_str, fontsize=15)
            plt.clim(lower, upper)
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel(uncertainty_str, rotation=270, fontsize=15)
            if scan_id is None:
                plt.savefig(f"{img_path}{name}_img{suffix}.png")
            else:
                plt.savefig(
                    f"{img_path}{name}_img{suffix}" + "_" + str(scan_id) + ".png"
                )
            plt.clf()

        # Plot histogram of NLL values across pixels
        nll_flat = nlls.flatten().detach().cpu().numpy()
        plt.hist(nll_flat, range=(-5, 20), density=True)
        plt.ylabel(f"Histogram Density", fontsize=15)
        plt.xlabel("NLL", fontsize=15)
        plt.title(f"{view_str}, {model_str}", fontsize=15)
        plt.yscale("log", nonposy="clip")
        plt.ylim((1e-5, 0.5))

        if scan_id is None:
            plt.savefig(f"{img_path}nll_hist{suffix}.png")
        else:
            plt.savefig(f"{img_path}nll_hist{suffix}" + "_" + str(scan_id) + ".png")
        plt.clf()

    if save_model_out:
        output = {}
        output["image_resolution"] = image_resolution
        output["gt"] = gt.detach().cpu()
        try:
            model_output["model_in"] = model_output["model_in"].detach().cpu()
        except:
            pass
        model_output["model_out"] = model_output["model_out"].detach().cpu()
        model_output["var_out"] = model_output["var_out"].detach().cpu()
        model_output["mean_out"] = model_output["mean_out"].detach().cpu()
        output["model_output"] = model_output
        output["args"] = args
        output["scan_id"] = scan_id
        with open(model_out_path + "model_out.pkl", "wb") as file:
            pickle.dump(output, file)
