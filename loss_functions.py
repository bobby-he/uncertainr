"""
Loss functions and accompanying utility functions, both for training on Sinograms
And comparing predictions to Ground Truth.
"""
import torch

import numpy as np
import pickle
import torchvision

# small constant to avoid numerical errors
EPS = 1e-6


def normal_nll(diffs, vars, mask=None, EPS=EPS):
    if mask is not None:
        diffs = diffs[mask]
        vars = vars[mask]
    t1 = 1 / 2 * torch.log(2 * np.pi * (vars + EPS)).mean()
    t2 = 1 / 2 * (torch.exp(torch.log(diffs**2 + EPS) - torch.log(vars + EPS))).mean()
    return t1 + t2


def image_mse(mask, model_output, gt):
    if mask is None:
        return {"img_loss": ((model_output["model_out"] - gt["img"]) ** 2).mean()}
    else:
        return {
            "img_loss": (mask * (model_output["model_out"] - gt["img"]) ** 2).mean()
        }


class single_image_mse_project:
    """Perform Sinogram projection and compute loss to ground truth."""

    def __init__(
        self,
        gt_coords,
        gt_img,
        thetas,
        theta_batch_size,
        projection_length,
        noise_sigma=0.001,
        key=0,
        regularize=None,
        reg_coeff=0,
        zero_noise=False,
    ):
        self.rng = np.random.default_rng(key)
        self.thetas = thetas
        self.theta_batch_size = theta_batch_size
        self.regularize = regularize
        self.projection_length = projection_length
        self.reg_coeff = reg_coeff
        self.noise_sigma = noise_sigma  # STD for noise generation

        # Compute and store Sinogram and Sinogram noise
        self.gt_project = {}
        self.thetas = thetas
        for theta in thetas:
            self.gt_project[theta] = ct_project(gt_coords, gt_img, theta)

        self.noise = {}
        for theta in thetas:
            self.noise[theta] = torch.from_numpy(
                self.rng.normal(scale=noise_sigma, size=(projection_length, 1))
            )
            if zero_noise:
                self.noise[theta] *= 0
            if torch.cuda.is_available():
                self.noise[theta] = self.noise[theta].to("cuda")

    def __call__(self, model_output):
        batch_thetas = self.rng.choice(
            self.thetas, self.theta_batch_size, replace=False
        )
        model_in = model_output["model_in"]

        # Calculate projected difference to noisy Sinogram
        diffs = torch.cat(
            [
                self.gt_project[theta]
                - ct_project(model_in, model_output["model_out"], theta)
                + self.noise[theta]
                for theta in batch_thetas
            ]
        )

        # Compute regularization.
        regularization = 0
        if self.regularize is not None:
            img = model_output["model_out"].reshape(
                (self.projection_length, self.projection_length)
            )
            regularization += compute_regularization(img, self.regularize)
            regularization = self.reg_coeff * regularization

        noise_vars = torch.ones(diffs.size()).to("cuda")
        if self.noise_sigma > 0:
            noise_vars *= self.noise_sigma**2

        loss_dct = {
            "proj_loss": normal_nll(diffs, noise_vars, EPS=5e-9) + regularization
        }
        return loss_dct


def image_mse_uncertain(
    model_output,
    gt,
    coverage_mse=False,
    C_save_path=None,
    scan_id=0,
):
    """Evaluate model_output compared to ground truth, gt, on prective accuracy and uncertainty metrics."""
    mean_preds = model_output["mean_out"]
    var_preds = model_output["var_out"]
    gt_img = gt["img"]
    results_dct = {}
    mse = ((mean_preds - gt_img) ** 2).mean()

    percent_diff = 0.05
    percentages = [round(percent, 2) for percent in np.arange(0, 1, percent_diff)]

    # Find optimal delta value. Delta is a small positive number used to widen predictive intervals
    # In order to account for 0 pixels (unattainable by Softmax-outputted models, like UncertaINR)
    # At the cost of slightly wider predictions. We search for delta over a grid of values.
    deltas = np.append(np.logspace(-20, -1, 21), 0)
    delta_vals = []
    best_delta = 0
    best_ece = np.inf
    for delta in deltas:
        ece = 0
        for percent in percentages:
            lb = torch.quantile(
                model_output["model_out"], 0.5 - percent / 2, dim=0, keepdim=True
            )
            ub = torch.quantile(
                model_output["model_out"], 0.5 + percent / 2, dim=0, keepdim=True
            )
            mean_pt = ((gt_img > lb - delta) & (gt_img < ub + delta)).float().mean()
            ece += abs(mean_pt - percent) * percent_diff
        delta_vals.append(ece)
        if ece < best_ece:
            best_delta = delta
            best_ece = ece

    # Using best_delta to calulcate and save final values for coverage and ECE
    # Also make plot of reliability curves and ECE per pixel
    cvg_sqd_diff = 0
    ece = 0
    ece_per_pixel = torch.zeros(model_output["model_out"].size()).cpu()
    c_vals = {}
    c_vals_no_delta = {}
    c_vals_per_pixel = {}
    for percent in percentages:
        lb = torch.quantile(
            model_output["model_out"], 0.5 - percent / 2, dim=0, keepdim=True
        )
        ub = torch.quantile(
            model_output["model_out"], 0.5 + percent / 2, dim=0, keepdim=True
        )
        mean_pt = (
            ((gt_img > lb - best_delta) & (gt_img < ub + best_delta)).float().mean()
        )
        mean_pt_no_delta = ((gt_img > lb) & (gt_img < ub)).float().mean()
        mean_per_pixel = (
            ((gt_img > lb - best_delta) & (gt_img < ub)).float().mean(dim=0)
        )

        cvg_sqd_diff += (mean_pt - percent) ** 2
        ece += abs(mean_pt - percent) * percent_diff
        ece_per_pixel += abs(mean_per_pixel.cpu() - percent) * percent_diff

        c_vals[percent] = mean_pt.cpu()
        c_vals_no_delta[percent] = mean_pt_no_delta.cpu()
        c_vals_per_pixel[percent] = mean_per_pixel.cpu()
    if coverage_mse:
        results_dct.update({"coverage_mse": cvg_sqd_diff / len(percentages)})
    if C_save_path is not None:
        with open(C_save_path + "C_vals_" + str(scan_id) + ".pkl", "wb") as file:
            pickle.dump(c_vals, file)
        with open(
            C_save_path + "C_vals_no_delta_" + str(scan_id) + ".pkl", "wb"
        ) as file:
            pickle.dump(c_vals_no_delta, file)
        with open(
            C_save_path + "C_vals_per_pixel_" + str(scan_id) + ".pkl", "wb"
        ) as file:
            pickle.dump(c_vals_per_pixel, file)
        with open(C_save_path + "ece_per_pixel_" + str(scan_id) + ".pkl", "wb") as file:
            pickle.dump(ece_per_pixel, file)

    diffs = mean_preds - gt_img
    results_dct.update(
        {
            "mean_pred_img_loss": mse,
            "nll": normal_nll(diffs, var_preds, None),
            "psnr": -10 * torch.log10(mse),
            "snr": torch.as_tensor([get_SNR(gt_img, mean_preds)]),
            "ece_val": ece,
            "ece_best_delta": best_delta,
        }
    )

    return results_dct


def get_SNR_stdev(noise_dB, gt_coords, gt, thetas):
    """Compute standard deviation for Gaussian noise corresponding to a given SNR in Sinogram"""
    sinogram = torch.cat([ct_project(gt_coords, gt, theta) for theta in thetas])
    N = torch.numel(sinogram)
    mu = torch.sum(sinogram**2) ** (0.5)
    sigma = mu / (10 ** (noise_dB / 20))
    sigma = sigma / (N ** (0.5))
    return sigma.item()


def get_SNR(gt, img):
    mu = torch.sum(gt**2) ** (0.5)
    sigma = torch.sum((gt - img) ** 2) ** (0.5)
    return 20 * torch.log10(mu / sigma).item()


def compute_regularization(img, reg_type="ISO_TV"):
    width, _ = img.size()
    reg_val = 0

    # isotropic implementation
    if reg_type == "ISO_TV":
        tv_h = ((img[1:, :] - img[:-1, :]).pow(2)).sum()
        tv_w = ((img[:, 1:] - img[:, :-1]).pow(2)).sum()
        reg_val = tv_h + tv_w

    # full isotropic implementation
    elif reg_type == "ISO_SQRT_TV":
        tv_h = (img[1:, :] - img[:-1, :]).pow(2)
        tv_w = (img[:, 1:] - img[:, :-1]).pow(2)
        tv = torch.sqrt(tv_h + tv_w)
        reg_val = tv.sum()

    # anisotropic approximation
    elif reg_type == "ANISO_TV":
        tv_h = ((img[1:, :] - img[:-1, :]).abs()).sum()
        tv_w = ((img[:, 1:] - img[:, :-1]).abs()).sum()
        reg_val = tv_h + tv_w

    # Huber loss
    elif reg_type == "HUBER":
        delta = 1
        tv_h = img[1:, :] - img[:-1, :]
        tv_temp = torch.clone(tv_h)
        tv_h[tv_temp <= delta] = 0.5 * tv_h.pow(2)
        tv_h[tv_temp > delta] = delta * (tv_h.pow(2) - 0.5 * delta)

        tv_w = img[:, 1:] - img[:, :-1]
        tv_temp = torch.clone(tv_w)
        tv_w[tv_temp <= delta] = 0.5 * tv_w.pow(2)
        tv_w[tv_temp > delta] = delta * (tv_w.pow(2) - 0.5 * delta)
        reg_val = tv_h.sum() + tv_w.sum()
    else:
        raise ValueError(f"Regularization {reg_type} not found")
    return reg_val / width


def ct_project(coords, img, theta, agg="mean"):
    """Rotates and sums up an image matrix by angle theta to return vector of projections."""
    sidelength = int(np.sqrt(max(img.size())))
    resampled = torchvision.transforms.functional.rotate(
        img.reshape(1, sidelength, sidelength),
        theta * 180 / np.pi,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        fill=0.0,
    )[0, :, :]
    if agg == "mean":
        weights = 1 / sidelength * torch.ones_like(resampled)
    elif agg == "variance":
        weights = 1 / sidelength**2 * torch.ones_like(resampled)
    return (resampled * weights).sum(axis=0)[:, None, ...]


def map_coordinates(input, coordinates):
    """PyTorch version of scipy.ndimage.interpolation.map_coordinates.
    Taken from https://github.com/sunset1995/pytorch-layoutnet.
    input: (H, W)
    coordinates: (2, ...)
    """
    h = input.shape[0]
    w = input.shape[1]

    def _coordinates_pad_wrap(h, w, coordinates):
        coordinates[0] = coordinates[0] % h
        coordinates[1] = coordinates[1] % w
        return coordinates

    co_floor = torch.floor(coordinates).long()
    co_ceil = torch.ceil(coordinates).long()
    d1 = coordinates[1] - co_floor[1].float()
    d2 = coordinates[0] - co_floor[0].float()
    print(co_floor)
    co_floor = _coordinates_pad_wrap(h, w, co_floor)
    co_ceil = _coordinates_pad_wrap(h, w, co_ceil)
    f00 = input[co_floor[0], co_floor[1]]
    f10 = input[co_floor[0], co_ceil[1]]
    f01 = input[co_ceil[0], co_floor[1]]
    f11 = input[co_ceil[0], co_ceil[1]]
    fx1 = f00 + d1 * (f10 - f00)
    fx2 = f01 + d1 * (f11 - f01)
    return fx1 + d2 * (fx2 - fx1)
