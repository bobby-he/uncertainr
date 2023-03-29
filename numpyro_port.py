# Utils for port from PyTorch to NumPyro for HMC
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.ndimage import map_coordinates

import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution
from numpyro.infer import NUTS
from numpyro.infer.hmc import hmc
from numpyro.infer.util import initialize_model, init_to_uniform


EPS = 1e-9


class NUTSWithInit(NUTS):
    """NUTS wrapper which does not override init_params"""

    def __init__(
        self,
        model=None,
        potential_fn=None,
        kinetic_fn=None,
        step_size=1.0,
        inverse_mass_matrix=None,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
        target_accept_prob=0.8,
        trajectory_length=None,
        max_tree_depth=10,
        init_strategy=init_to_uniform,
        find_heuristic_step_size=False,
        forward_mode_differentiation=False,
        regularize_mass_matrix=True,
    ):
        super(NUTSWithInit, self).__init__(
            potential_fn=potential_fn,
            model=model,
            kinetic_fn=kinetic_fn,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            dense_mass=dense_mass,
            target_accept_prob=target_accept_prob,
            trajectory_length=trajectory_length,
            init_strategy=init_strategy,
            find_heuristic_step_size=find_heuristic_step_size,
            forward_mode_differentiation=forward_mode_differentiation,
            regularize_mass_matrix=regularize_mass_matrix,
            max_tree_depth=max_tree_depth,
        )

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            (
                new_init_params,
                potential_fn,
                postprocess_fn,
                model_trace,
            ) = initialize_model(
                rng_key,
                self._model,
                dynamic_args=True,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs,
                forward_mode_differentiation=self._forward_mode_differentiation,
            )
            if init_params is None:
                init_params = new_init_params
            if self._init_fn is None:
                self._init_fn, self._sample_fn = hmc(
                    potential_fn_gen=potential_fn,
                    kinetic_fn=self._kinetic_fn,
                    algo=self._algo,
                )
            self._potential_fn_gen = potential_fn
            self._postprocess_fn = postprocess_fn
        elif self._init_fn is None:
            self._init_fn, self._sample_fn = hmc(
                potential_fn=self._potential_fn,
                kinetic_fn=self._kinetic_fn,
                algo=self._algo,
            )

        return init_params


def ct_project(coords, img, theta):
    """JAX numpy version of CT Project"""
    sidelength = int(np.sqrt(max(img.shape)))  # Assume img is flattened.
    y, x = coords[..., 0], coords[..., 1]
    x_rot = x * jnp.cos(theta) - y * jnp.sin(theta)
    y_rot = x * jnp.sin(theta) + y * jnp.cos(theta)
    x_rot = (x_rot + 1).reshape(sidelength, sidelength) * (sidelength / 2)
    y_rot = (y_rot + 1).reshape(sidelength, sidelength) * (sidelength / 2)
    sample_coords = jnp.stack([y_rot, x_rot])
    resampled = map_coordinates(
        img.reshape(sidelength, sidelength), sample_coords, order=1
    ).reshape(sidelength, sidelength)
    return (resampled / sidelength).sum(axis=0)[:, None, ...]


class GaussianSinogramWithTV(Distribution):
    """Gaussian sinogram observation model"""

    def __init__(
        self,
        coords,
        model_output,
        thetas,
        projection_length,
        noise_sigma=0.001,
        regularize=None,
        reg_coeff=0,
        zero_noise=False,
        noise=None,
        temperature=1,
    ):
        super(GaussianSinogramWithTV, self).__init__(batch_shape=jnp.shape(1))
        self.rng_key = jax.random.PRNGKey(0)
        self.coords = coords
        self.thetas = thetas
        self.regularize = regularize
        self.projection_length = projection_length
        self.reg_coeff = reg_coeff

        self.noise_sigma = noise_sigma
        if noise is not None:
            self.noise = noise
        else:
            self.noise = noise_sigma * random.normal(
                self.rng_key, shape=(len(thetas), projection_length, 1)
            )
        if zero_noise:
            self.noise *= 0

        self.model_project = jax.vmap(ct_project, (None, None, 0))(
            coords, model_output, thetas
        )
        self.model_output = model_output
        self.temperature = temperature

    def log_prob(self, gt_project):
        """
        Gaussian Sinogram log probability
        """
        # Model differences with ground truth observations in sinogram space, with noise.
        diffs = self.model_project + self.noise - gt_project

        # Add regularisation
        regularization = 0
        if self.regularize is not None:
            img = self.model_output.reshape(
                self.projection_length, self.projection_length
            )
            regularization += compute_regularization(img, self.regularize)
            regularization *= self.reg_coeff

        log_prob = -1 / 2 * ((diffs / self.noise_sigma) ** 2).sum()  # Gaussian log prob

        # Multiply regularisation by size of projection, because we need log prob sum not log prob mean
        regularized_logp = log_prob - regularization * jnp.size(diffs)
        return regularized_logp / self.temperature


def compute_regularization(img, reg_type="ISO_TV"):
    width, _ = img.shape
    reg_val = 0

    # Isotropic implementation
    if reg_type == "ISO_TV":
        tv_h = ((img[1:, :] - img[:-1, :]).pow(2)).sum()
        tv_w = ((img[:, 1:] - img[:, :-1]).pow(2)).sum()
        reg_val = tv_h + tv_w

    # Full isotropic implementation
    elif reg_type == "ISO_SQRT_TV":
        tv_h = (img[1:, :] - img[:-1, :]).pow(2)
        tv_w = (img[:, 1:] - img[:, :-1]).pow(2)
        tv = jnp.sqrt(tv_h + tv_w)
        reg_val = tv.sum()

    # Anisotropic approximation
    elif reg_type == "ANISO_TV":
        tv_h = jnp.abs(img[1:, :] - img[:-1, :]).sum()
        tv_w = jnp.abs(img[:, 1:] - img[:, :-1]).sum()
        reg_val = tv_h + tv_w
    else:
        raise ValueError(f"Regularization {reg_type} not found")
    return reg_val / width


def create_numpyro_model(
    coords,
    torch_module,
    gt_project,
    gt_coords,
    thetas,
    padded_sidelength,
    noise_sigma,
    noise,
    args,
):
    """Port torch model into numpyro"""
    assert len(coords.shape) == 3
    N = coords.shape[1]
    params = {}

    # Create forward model
    if args.model.name == "mlp":
        assert args.model.model_type == "rff_enc"
        avals = torch_module.positional_encoding.avals.cpu().numpy()
        bvals = numpyro.deterministic(
            "bvals", torch_module.positional_encoding.bvals.cpu().numpy()
        )

        out = jnp.concatenate(
            [
                avals * jnp.sin(coords[0] @ bvals.T + jnp.pi / 4),
                avals * jnp.cos(coords[0] @ bvals.T + jnp.pi / 4),
            ],
            axis=-1,
        )
        nonlin_dict = {
            "relu": jax.nn.relu,
            "sigmoid": jax.nn.sigmoid,
            "tanh": jnp.tanh,
            "selu": jax.nn.selu,
            "silu": jax.nn.silu,
            "softplus": jax.nn.softplus,
            "elu": jax.nn.elu,
        }
        nl = nonlin_dict[args.model.activation_type]
        for id, layer in enumerate(torch_module.net._net):
            if id != 0:
                out = nl(out)

            lin_layer = layer[0]
            out_dim, in_dim = lin_layer.weight.shape
            sigma = 1 / jnp.sqrt(args.uncertainty.tau * in_dim)
            params[f"w{id}"] = numpyro.sample(
                f"w{id}",
                dist.Normal(
                    jnp.zeros((out_dim, in_dim)),
                    jnp.ones((out_dim, in_dim)) * sigma,
                ),
            )
            out = jnp.matmul(out, params[f"w{id}"].T)

            assert out.shape == (N, out_dim)
            if args.model.bias:
                params[f"b{id}"] = numpyro.sample(
                    f"b{id}",
                    dist.Normal(
                        jnp.zeros((1, out_dim)), jnp.ones((1, out_dim)) * sigma
                    ),
                )
                out = out + params[f"b{id}"]

    elif args.model.name == "grid_of_pixels":
        from numpyro.distributions import constraints

        sigma = 1 / jnp.sqrt(args.uncertainty.tau)

        params["gop"] = numpyro.sample(
            "gop", dist.ImproperUniform(constraints.real, (), event_shape=(N, 1))
        )
        out = params["gop"]
    else:
        raise NotImplementedError

    out = jax.nn.sigmoid(out)
    if args.data.zero_pad:
        output_width = int(np.sqrt(N))
        out = out.reshape((output_width, output_width))
        sidelength = args.data.sidelength
        padding = int(((2 * sidelength**2) ** 0.5 - sidelength) / 2) + 1
        out = jnp.pad(out, pad_width=padding)
    out = out.reshape((1, -1, 1))

    # Create observation model
    obs_dist = GaussianSinogramWithTV(
        gt_coords,
        out,
        thetas,
        projection_length=padded_sidelength,
        noise_sigma=noise_sigma,
        regularize=args.reg.type,
        reg_coeff=args.reg.coeff,
        zero_noise=True if args.data.name == "shepp_2d" else False,
        noise=noise,
        temperature=args.uncertainty.temp,
    )

    numpyro.sample("Pred_project", obs_dist, obs=gt_project)
    out = numpyro.deterministic("out", out)
