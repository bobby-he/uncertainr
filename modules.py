from torch import nn
from torch.nn import functional as F
from functools import partial
import torch
import numpy as np
import math
from collections import defaultdict


def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=False)


class UncertainFCBlock(nn.Module):
    """
    A fully connected neural network, determined by uncertainty_cfg config.
    """

    def __init__(
        self,
        uncertainty_cfg,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=False,
        nonlinearity="relu",
        omega_0=30,
        bias=True,
        zero_pad=0,
    ):
        super().__init__()

        self.first_layer_init = None

        self.zero_pad = zero_pad

        nls_and_inits = {
            "sine": (
                Sine(omega_0),
                partial(sine_init, omega_0=omega_0),
                partial(first_layer_sine_init),
            ),
            "relu": (nn.ReLU(inplace=True), init_relu, None),
            "sigmoid": (nn.Sigmoid(), init_xavier, None),
            "tanh": (nn.Tanh(), None, init_xavier, None),
            "selu": (nn.SELU(inplace=True), init_selu, None),
            "silu": (nn.SiLU(inplace=True), init_selu, None),
            "softplus": (nn.Softplus(), init_normal, None),
            "elu": (nn.ELU(inplace=True), init_elu, None),
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
        self._net = []
        self._net.append(
            nn.Sequential(nn.Linear(in_features, hidden_features, bias=bias), nl)
        )
        for i in range(num_hidden_layers):
            self._net.append(
                nn.Sequential(
                    nn.Linear(hidden_features, hidden_features, bias=bias), nl
                )
            )

        if outermost_linear:
            self._net.append(
                nn.Sequential(nn.Linear(hidden_features, out_features, bias=bias))
            )
        else:
            self._net.append(
                nn.Sequential(
                    nn.Linear(hidden_features, out_features, bias=bias), nn.Sigmoid()
                )
            )
        self.net = nn.Sequential(*self._net)
        self.net.apply(nl_weight_init)

        if first_layer_init is not None:  # Initialization for SIREN first layer.
            self.net[0].apply(first_layer_init)

        self.width = hidden_features
        self.uncertainty_cfg = uncertainty_cfg
        self.output_dim = out_features

    def forward(self, coords, sample=True):
        output_mean = coords
        for i, layer in enumerate(self.net):
            if self.uncertainty_cfg.pdrop > 0 and i > 0:
                output_mean = MC_dropout(
                    output_mean,
                    self.uncertainty_cfg.pdrop,
                    mask=self.training or sample,
                )
            output_mean = layer(output_mean)
            layer = None
            del layer

        if self.zero_pad > 0:
            output_width = int(np.sqrt(output_mean.size()[1]))
            padding = self.zero_pad
            output_mean = output_mean.reshape((output_width, output_width))
            output_mean = torch.nn.functional.pad(
                output_mean, (padding, padding, padding, padding)
            )
            output_mean = output_mean.reshape((1, -1, 1))
        return {"mean": output_mean}

    def sample_predict(self, coords, Nsamples):
        # Sample and aggregate over multiple forward passes
        predictions = defaultdict(list)

        for i in range(Nsamples):
            y = self.forward(coords, sample=True)
            for key in y:
                predictions[key].append(y[key])

        for key, val in predictions.items():
            if val[0] is not None:
                predictions[key] = torch.cat(val)
            else:
                predictions[key] = None
        return predictions


class UncertaINR(nn.Module):
    """ "Base UncertaINR model class"""

    def __init__(
        self,
        uncertainty_cfg,
        out_features=1,
        type="sine",
        in_features=2,
        mode="mlp",
        embed_width=1,
        hidden_features=256,
        num_hidden_layers=3,
        omega_0=30,
        bias=True,
        zero_pad=0,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.zero_pad = zero_pad
        if self.mode == "rbf":
            self.rbf_layer = RBFLayer(
                in_features=in_features,
                out_features=kwargs.get("rbf_centers", 1024),
                omega_0=omega_0,
            )
            in_features = kwargs.get("rbf_centers", 1024)
        elif self.mode == "nerf":
            self.positional_encoding = PosEncodingNeRF(
                in_features=in_features,
                sidelength=kwargs.get("sidelength", None),
                fn_samples=kwargs.get("fn_samples", None),
                use_nyquist=kwargs.get("use_nyquist", False),
            )
            in_features = self.positional_encoding.out_dim
        elif self.mode == "rff_enc":
            self.positional_encoding = PosEncodingRFF(
                in_features=in_features,
                out_dim=int(embed_width / 2),
                omega_0=omega_0,
                b_params=True,
            )
            in_features = self.positional_encoding.out_dim
        self.net = UncertainFCBlock(
            uncertainty_cfg,
            in_features=in_features,
            out_features=out_features,
            num_hidden_layers=num_hidden_layers,
            hidden_features=hidden_features,
            outermost_linear=kwargs.get("outermost_linear", True),
            nonlinearity=type,
            omega_0=omega_0,
            bias=bias,
            zero_pad=zero_pad,
        )
        self.uncertainty_cfg = uncertainty_cfg
        self.zero_pad = zero_pad
        print(self)

    def process_coords(self, coords):
        # Various input processing methods for different applications
        if self.mode == "rbf":
            coords = self.rbf_layer(coords)
        elif self.mode in ["nerf", "rff_enc"]:
            coords = self.positional_encoding(coords)
        return coords

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach()
        coords = self.process_coords(coords_org)
        output = self.net(coords)
        out_dct = {
            "model_in": coords_org,
            "model_out": output["mean"],
        }
        return out_dct

    def sample_predict(self, model_input, Nsamples):
        coords = model_input["coords"]
        processed_coords = self.process_coords(coords)
        output = self.net.sample_predict(processed_coords, Nsamples)
        return {
            "model_in": coords,
            "model_out": output["mean"],
        }


class Ensemble:
    """
    A container class for ensembles which holds a group of baselearners
    """

    def __init__(self, baselearners):
        self.baselearners = baselearners
        self.ensemble_size = len(self.baselearners)

    def sample_predict(self, model_input, Nsamples=1):
        # Sample and aggregate over multiple forward passes and baselearners
        coords = model_input["coords"]
        predictions = defaultdict(list)

        for _, bslr in enumerate(self.baselearners):
            bslr.cuda()
            num_samples = int(Nsamples / self.ensemble_size)
            num_samples = max(num_samples, 1)
            output = bslr.sample_predict(model_input, num_samples)
            for key in output:
                predictions[key].append(output[key])
            output = None
            del output
            bslr.cpu()
            del bslr
            torch.cuda.empty_cache()
        for key, val in predictions.items():
            if val[0] is not None:
                predictions[key] = torch.cat(val)
            else:
                predictions[key] = None
        predictions.update({"model_in": coords})
        return predictions


class PixelLookup(nn.Linear, nn.Module):
    """ "Returns weight parameters, for Grid of Pixels."""

    __doc__ = nn.Linear.__doc__

    def forward(self, dummy_input):
        return self.weight.T.unsqueeze(0)


class GridOfPixels(nn.Module):
    def __init__(self, in_features, out_features, zero_pad=0, pad_val=0):
        super().__init__()

        self.zero_pad = zero_pad
        self.pad_val = pad_val

        self.net = []
        self.net.append(
            nn.Sequential(
                PixelLookup(in_features, out_features, bias=False), nn.Sigmoid()
            )
        )

        self.net = nn.Sequential(*self.net)
        self.net.apply(init_weights_gop)

    def forward(self, input, **kwargs):
        coords = input["coords"].clone().detach()

        output = self.net(coords)  # Input is dummy in GOP
        if self.zero_pad > 0:
            output_width = int(np.sqrt(output.size()[1]))
            padding = self.zero_pad
            output = output.reshape((output_width, output_width))
            output = torch.nn.functional.pad(
                output, (padding, padding, padding, padding), value=self.pad_val
            )
            output = output.reshape((1, -1, 1))

        return {"model_in": coords, "model_out": output}


def init_relu(m):
    if hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


def init_normal(m):
    if hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")


def init_selu(m):
    if hasattr(m, "weight"):
        num_input = m.weight.size(-1)
        nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_elu(m):
    if hasattr(m, "weight"):
        num_input = m.weight.size(-1)
        nn.init.normal_(
            m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input)
        )


def init_xavier(m):
    if hasattr(m, "weight"):
        nn.init.xavier_normal_(m.weight)


def init_weights_gop(m):
    if hasattr(m, "weight"):
        nn.init.uniform_(m.weight, a=-1.0, b=-1.0)


def sine_init(m, omega_0=30):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / omega_0, np.sqrt(6 / num_input) / omega_0
            )


def first_layer_sine_init(m, scale=1):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-scale / num_input, scale / num_input)


class Sine(nn.Module):
    # See supplement Sec. 1.5 of SIREN paper for discussion of default frequency 30
    def __init__(self, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, input):
        return torch.sin(self.omega_0 * input)


class PosEncodingNeRF(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            self.num_frequencies = 4
            if use_nyquist:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = self.get_num_frequencies_nyquist(
                    min(sidelength[0], sidelength[1])
                )
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        in_dim = coords.ndim
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2**i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2**i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        if in_dim == 2:
            return coords_pos_enc.reshape(coords.shape[0], self.out_dim)
        else:
            return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class PosEncodingRFF(nn.Module):
    """Module to add RFF encoding as in [Tancik et al. 2020]."""

    def __init__(self, in_features, out_dim, omega_0=1, b_params=True):
        super().__init__()

        self.in_features = in_features
        if b_params:
            self.bvals = nn.Parameter(
                omega_0 * torch.Tensor(np.random.normal(size=(out_dim, in_features)))
            )
            self.bvals.requires_grad = False
        else:
            self.bvals = omega_0 * torch.Tensor(
                np.random.normal(size=(out_dim, in_features))
            )
            self.bvals = self.bvals.cuda()  # .requires_grad=False

        self.avals = torch.Tensor(np.ones(out_dim)).cuda()
        self.out_dim = 2 * out_dim

    def forward(self, coords):
        in_dim = coords.ndim
        coords = coords.view(coords.shape[0], -1, self.in_features)

        # coords_pos_enc = coords
        coords_pos_enc = torch.cat(
            [
                self.avals * torch.sin((coords) @ self.bvals.T + np.pi / 4),
                self.avals * torch.cos((coords) @ self.bvals.T + np.pi / 4),
            ],
            axis=-1,
        )

        if in_dim == 2:
            return coords_pos_enc.reshape(coords.shape[0], self.out_dim)
        else:
            return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class RBFLayer(nn.Module):
    """Transforms incoming data using a given radial basis function.
    - Input: (1, N, in_features) where N is an arbitrary batch size
    - Output: (1, N, out_features) where N is an arbitrary batch size"""

    def __init__(self, in_features, out_features, omega_0=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.omega_0 = omega_0
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, self.omega_0)

    def forward(self, input):
        if input.ndim == 3:
            input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = torch.sqrt((x - c).pow(2).sum(-1)) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2) / 2)
        return phi
