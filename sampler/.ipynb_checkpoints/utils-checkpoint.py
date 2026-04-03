import torch
import numpy as np
import abc
import functools
from tqdm import tqdm
from scipy import integrate
from sampler.sde import VESDE, VPSDE
from trainer.loss import predict_fn, voriticity_residual, sample_noise, kse_residual
from einops import rearrange
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import Voronoi, cKDTree


class ODE_Solver(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, net_fn, eps=None):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        if sde.config.sde != 'poisson':
            self.rsde = sde.reverse(net_fn, probability_flow=True)
        self.net_fn = net_fn
        self.eps = eps

    @abc.abstractmethod
    def update_fn(self, x, t, t_list=None, idx=None):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__()
        self.sde = sde
        self.channel_modal = channel_modal
        # Compute the reverse SDE/ODE
        if sde.config.sde != 'poisson':
            self.rsde = sde.reverse(net_fn, probability_flow)
        self.net_fn = net_fn
        self.eps = eps

    @abc.abstractmethod
    def update_fn(self, x, t, t_list=None, idx=None):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        super().__init__()
        self.sde = sde
        self.net_fn = net_fn
        self.snr = snr
        self.n_steps = n_steps
        self.channel_modal = channel_modal

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, eps=None):
        super().__init__(sde, net_fn, probability_flow, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        z = torch.randn_like(x)
        if self.sde.config.sde == 'poisson':
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
                dt = float(dt.cpu().numpy())
            drift = self.sde.ode(self.net_fn, x, t)
            diffusion = torch.zeros((len(x)), device=x.device)
        else:
            if t_list is None:
                dt = -1. / self.sde.N
            drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ForwardEulerPredictor(ODE_Solver):
    def __init__(self, sde, net_fn, eps=None):
        super().__init__(sde, net_fn, eps)

    def update_fn(self, x, t, t_list=None, idx=None):

        if self.sde.config.sde == 'poisson':
            # dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            drift = self.sde.ode(self.net_fn, x, t)
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
                dt = float(dt.cpu().numpy())
        else:
            dt = -1. / self.sde.N
            drift, _ = self.rsde.sde(x, t)
        x = x + drift * dt
        return x


class ImprovedEulerPredictor(ODE_Solver):
    def __init__(self, sde, net_fn, eps=None):
        super().__init__(sde, net_fn, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        if self.sde.config.sde == 'poisson':
            if t_list is None:
                dt = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
            else:
                # integration over z
                dt = (torch.exp(t_list[idx + 1] - t_list[idx]) - 1)
                dt = float(dt.cpu().numpy())
            drift = self.sde.ode(self.net_fn, x, t)
        else:
            dt = -1. / self.sde.N
            drift, _ = self.rsde.sde(x, t)
        x_new = x + drift * dt

        if idx == self.sde.N - 1:
            return x_new
        else:
            idx_new = idx + 1
            t_new = t_list[idx_new]
            t_new = torch.ones(len(t), device=t.device) * t_new

            if self.sde.config.sde == 'poisson':
                if t_list is None:
                    dt_new = - (np.log(self.sde.config.z_max) - np.log(self.eps)) / self.sde.N
                else:
                    # integration over z
                    dt_new = (1 - torch.exp(t_list[idx] - t_list[idx + 1]))
                    dt_new = float(dt_new.cpu().numpy())
                drift_new = self.sde.ode(self.net_fn, x_new, t_new)
            else:
                drift_new, diffusion = self.rsde.sde(x_new, t_new)
                dt_new = -1. / self.sde.N

            x = x + (0.5 * drift * dt + 0.5 * drift_new * dt_new)
            return x


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__(sde, net_fn, probability_flow, channel_modal, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        t_shape = t.shape
        f, G = self.rsde.discretize(x, t, self.channel_modal)
        z = torch.randn_like(x)
        x_mean = x - f
        G = G.view(*t_shape)
        if self.channel_modal is None:
            x = x_mean + G[:, None, None, None] * z
        else:
            G = G.repeat_interleave(torch.tensor(self.channel_modal).to(G.device), dim=1)
            x = x_mean + G[:, :, None, None] * z
        return x, x_mean


class ReverseDiffusionPredictorMM(Predictor):
    def __init__(self, sde, net_fn, probability_flow=False, channel_modal=None, eps=None):
        super().__init__(sde, net_fn, probability_flow, channel_modal, eps)

    def update_fn(self, x, t, t_list=None, idx=None):
        t_shape = t.shape
        f, G = self.rsde.discretize(x, t, self.channel_modal)
        z = sample_noise(x.shape, channel_modal=self.channel_modal, device=x.device, dtype=x.dtype)
        x_mean = x - f
        G = G.view(*t_shape)
        if self.channel_modal is None:
            x = x_mean + G[:, None, None, None] * z
        else:
            G = G.repeat_interleave(torch.tensor(self.channel_modal).to(G.device), dim=1)
            x = x_mean + G[:, :, None, None] * z
        return x, x_mean


class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, net_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, t_list=None, idx=None):
        return x, x



class LangevinCorrector(Corrector):
  def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
    super().__init__(sde, net_fn, snr, n_steps, channel_modal)
    if not isinstance(sde, VPSDE) \
        and not isinstance(sde, VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    net_fn = self.net_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, VPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones(len(x)).float().to(t.device)

    for i in range(n_steps):
      grad = net_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      if len(x.shape) > 4:
        x_mean = x + step_size[:, None, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None, None] * noise
      else:
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


class LangevinCorrectorMM(Corrector):
    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        super().__init__(sde, net_fn, snr, n_steps, channel_modal)
        if not isinstance(sde, VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        net_fn = self.net_fn
        n_steps = self.n_steps
        target_snr = self.snr

        if isinstance(sde, VESDE):
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = net_fn(x, t)
            noise = sample_noise(x.shape, channel_modal=self.channel_modal, device=x.device, dtype=x.dtype)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            if self.channel_modal is None:
                x_mean = x + step_size[:, None, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
            else:
                step_size = step_size.repeat_interleave(torch.tensor(self.channel_modal).to(step_size.device), dim=1)
                x_mean = x + step_size[:, :, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, :, None, None] * noise

        return x, x_mean


class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, net_fn, snr, n_steps):
        super().__init__(sde, net_fn, snr, n_steps)
        if not isinstance(sde, VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        net_fn = self.net_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VESDE):
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = net_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, net_fn, snr, n_steps, channel_modal=None):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_ode_solver_update_fn(x, t, sde, net, ode_solver, eps, t_list=None, idx=None):
    """A wrapper that configures and returns the update function of ODE solvers."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    ode_solver_obj = ode_solver(sde, net_fn, eps)
    return ode_solver_obj.update_fn(x, t, t_list=t_list, idx=idx)


def shared_predictor_update_fn(x, t, sde, net, predictor, probability_flow, continuous, eps,
                               channel_modal=None, t_list=None, idx=None):
    """A wrapper that configures and returns the update function of predictors."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, net_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, net_fn, probability_flow, channel_modal, eps)
    return predictor_obj.update_fn(x, t, t_list=t_list, idx=idx)


def shared_corrector_update_fn(x, t, sde, net, corrector, continuous, snr, n_steps, channel_modal=None):
    """A wrapper tha configures and returns the update function of correctors."""
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, net_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, net_fn, snr, n_steps, channel_modal=channel_modal)
    return corrector_obj.update_fn(x, t)


def ode_sampler(net, sde, ode_solver, shape, device='cpu', dtype='float32', eps=1e-3):
    ode_update_fn = functools.partial(shared_ode_solver_update_fn,
                                      sde=sde,
                                      ode_solver=ode_solver,
                                      eps=eps)
    x = sde.prior_sampling(shape).to(device).float()
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()

    xs = []
    for i in tqdm(range(sde.N), desc='generating...', total=sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device).float() * t
        x = ode_update_fn(x, vec_t, net=net, t_list=timesteps, idx=i)
        xs.append(x)
    return x, sde.N


def pc_sampler(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous,
                                            eps=eps,
                                            channel_modal=config.channel_modal)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            channel_modal=config.channel_modal)
    # if is_mm:
    #     x = sample_noise(shape, channel_modal=config.channel_modal, device=device, dtype=dtype_torch)*sde.sigma_max
    # else:
    x = sde.prior_sampling(shape).to(device).float()
    x0 = torch.tensor(x0, device=device).float()
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    # if 'mm' in config.version:
    #     mode = np.array(config.mm_mode)
    #     mode_r = torch.tensor(mode.repeat(config.channel_modal, 0), device=device).bool()

    x_generated = [x.detach().cpu().numpy()]
    for i in tqdm(range(sde.N)):
        t = timesteps[i]
        # if 'mm' in config.version:
        #     t = torch.tensor(util.mode_to_ts(mode, pos=eps_t, neg=t), device=device).float()
        #     vec_t = torch.ones([shape[0], config.num_modals], device=t.device).float() * t[None, :]
        #     x = x * (~mode_r)[None, :, None, None] + x0 * mode_r[None, :, None, None]
        #     if 'cond' in config.version:
        #         vec_t = torch.cat([vec_t, torch.ones_like(vec_t[:, :1])*pattern], dim=1)
        # else:
        vec_t = torch.ones(shape[0], device=t.device).float() * t
        x, x_mean = corrector_update_fn(x, vec_t, net=net)
        # if 'mm' in config.version:
        #     x = x * (~mode_r)[None, :, None, None] + x0 * mode_r[None, :, None, None]
        x, x_mean = predictor_update_fn(x, vec_t, net=net)
        x_generated.append(x_mean.detach().cpu().numpy())

    return x_mean if denoise else x, x_generated


def generate_ar_2d(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1., mult=10, transform_init=None, num_steps=10, overlap=1,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            )
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(x0), nf, ncomp+config.num_conditions, config.image_size, config.image_size]       # batch*nf*(c+npara)*h*w

    transform_ar, alpha_ar = lambda x: x[:, :ol], alpha
    transform_init, alpha_init = transform_ar if transform_init is None else transform_init, alpha*mult

    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
            transform = transform_init
            alpha = alpha_init
        else:
            y = x_mean[:, -ol:].detach()
            transform = transform_ar
            alpha = alpha_ar
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                with torch.no_grad():
                    f, G = sde.discretize(x, vec_t)
                    rev_f = f - G[:, None, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    x_mean = x - rev_f
                    x_u = x_mean + rev_G[:, None, None, None, None] * z
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                loss = alpha * loss_dps
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated



def generate_ar_1d(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1., mult=10, transform_init=None, num_steps=10, overlap=1,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,)
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [config.num_samples, nf, ncomp+config.num_conditions, config.image_size]       # batch*nf*(c+npara)*h

    transform_ar, alpha_ar = lambda x: x[:, :ol], alpha
    transform_init, alpha_init = transform_ar if transform_init is None else transform_init, alpha*mult

    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
            transform = transform_init
            alpha = alpha_init
        else:
            y = x_mean[:, -ol:].detach()
            transform = transform_ar
            alpha = alpha_ar
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                with torch.no_grad():
                    f, G = sde.discretize(x, vec_t)
                    rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    x_mean = x - rev_f
                    x_u = x_mean + rev_G[:, None, None, None] * z
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                loss = alpha * loss_dps
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated


def s3gm_sample_2d(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                T_prime_y=10, T_prime=0, overlap=1,
                              device='cpu', dtype='float32', eps=1e-12, 
                              probability_flow=False, continuous=True):
    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_2d(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                    probability_flow=probability_flow, continuous=continuous)
    else:
        x_y = y
    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = generate_ar_2d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0, 
                                    n_steps=n_steps, probability_flow=probability_flow, 
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
    return np.concatenate([x_y, x_extra[:, overlap:]], axis=1) if T_prime > x_y.shape[1] else x_y


def s3gm_sample_1d(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                T_prime_y=10, T_prime=0, overlap=1,
                              device='cpu', dtype='float32', eps=1e-12, 
                              probability_flow=False, continuous=True):
    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_1d(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                    probability_flow=probability_flow, continuous=continuous)
    else:
        x_y = y
    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = generate_ar_1d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0, 
                                    n_steps=n_steps, probability_flow=probability_flow, 
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
    return np.concatenate([x_y, x_extra[:, overlap:]], axis=1) if T_prime > x_y.shape[1] else x_y


def generate_parallel_2d(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, gamma=1.e-2,
                          num_steps=10, overlap=1,
                              device='cpu', dtype='float32', eps=1e-3, save_sample_path=False,
                              probability_flow=False, continuous=True, data_scalar=None):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,)
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    # x_known = torch.from_numpy(x0).to(device).type(dtype_torch)
    y = torch.from_numpy(y).to(device).type(dtype_torch)
    # shape_sample = [len(y), config.num_channels, config.image_size, config.image_size]

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    nc = config.num_conditions
    shape = [len(y), b, nf, ncomp+nc, config.image_size, config.image_size]       # batch*b*nf*(c+npara)*h*w
    # shape_sample = [config.num_samples, ns_real, ncomp+nc, config.image_size, config.image_size]     # batch*ns_real*(c+npara)*h*w

    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([config.num_samples, ns_real, ncomp+nc, config.image_size, config.image_size], dtype=dtype_torch, device=device)   # batch*ns_real*(c+npara)*h*w
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        # sample[:, :, ncomp:] = xx[:, 0, 0:1, ncomp:]
        return sample

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    x_unknown = sde.prior_sampling(shape).to(device).float()    # batch*b*nf*(c+npara)*h*w
    # x_u_temp = sde.prior_sampling(shape_sample)      # batch*ns_real*(c+npara)*h*w
    # x_unknown = []
    # for i in range(b):
    #     x_unknown.append(x_u_temp[:, i*(nf-ol):i*(nf-ol)+nf])
    # x_unknown = torch.stack(x_unknown, dim=1).float().to(device)
    
    x = x_unknown           # batch*b*(nf*c+npara)*h*w
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[])
    with tqdm(range(sde.N)) as tqdm_setting:
        for i in range(sde.N):
            t = timesteps[i]

            '''method 1 (batched)'''
            vec_t = torch.ones(shape[0]*b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')       # (batch*b)*nf*(c+npara)*h*w

            '''corrector'''
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)     # (batch*b)*nf*(c+npara)*h*w

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h w -> (b n) t c h w')

            with torch.enable_grad():
                inp = temp.clone()                  # (batch*b)*nf*(c+npara)*h*w
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)          # (batch*b)*nf*(c+npara)*h*w
                with torch.no_grad():
                    f, G = sde.discretize(temp, vec_t)
                    rev_f = f - G[:, None, None, None, None] ** 2 * score.detach() * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    temp_mean = temp - rev_f
                    temp_u = temp_mean + rev_G[:, None, None, None, None] * zb
                    #print(rev_f)
                    #print(rev_G)
                # dps loss
                _, std = sde.marginal_prob(xb, vec_t)
                if isinstance(sde, VESDE):
                    x0_hat = rearrange(std[:, None, None, None, None] ** 2 * score + inp, '(b n) t c h w -> b n t c h w', n=b)     # batch*b*nf*(c+npara)*h*w
                else:
                    alpha_sqrt_ = (1-std**2).sqrt()[:, None, None, None, None]
                    x0_hat = rearrange((std[:, None, None, None, None] ** 2 * score + inp)/alpha_sqrt_, '(b n) t c h w -> b n t c h w', n=b)     # batch*b*nf*(c+npara)*h*w
                x0_hat_temp = x_to_sample(x0_hat)
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())

                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_temp)) ** 2 / var).reshape(x0_hat.shape[0], -1), dim=-1)  # /scalar.sqrt()  *std[None, :, None, None]
                loss_dps = torch.sum(loss_dps, dim=0)  # /loss_dps.detach().mean().sqrt()
                if std_y is not None:
                    loss_dps = loss_dps/2.
                # loss_dps = loss_dps/loss_dps.detach().sqrt()    # normalize

                loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)    # , len(x0_hat[0])
                loss_consis = torch.sum(loss_consis, dim=-1).mean()        # *torch.softmax(loss_consis.detach(), dim=1)
                # loss_consis = loss_consis/loss_consis.detach().sqrt()    # normalize
                # loss_consis_para = torch.sum(((x0_hat[:, 1:, :, ncomp:]-x0_hat[:, 0:1, :, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_consis_para = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, ncomp:].detach()-x0_hat[:, 1:, :ol, ncomp:])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()
                # loss_consis_para = loss_consis_para/loss_consis_para.detach().sqrt()    # normalize

                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e}')
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())

                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                temp = temp_u - dx     # (batch*b)*(nf*c+npara)*h*w
            #     # x = x_u
            temp = temp.detach()
            #print(temp.shape)
            x = rearrange(temp, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(temp_mean, '(b n) t c h w -> b n t c h w', n=b)
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            tqdm_setting.update(1)

    '''
    # ========================= 【只保存 loss_dps】 =========================
    try:
        import pandas as pd
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        csv_name = f"loss_dps_SDE_{timestamp}.csv"
        
        # 直接指定只存 loss_dps，简单粗暴，绝不报错
        data_to_save = {
            'step': range(1, len(losses['loss_dps']) + 1),
            'loss_dps': losses['loss_dps']
        }
        
        df_loss = pd.DataFrame(data_to_save)
        df_loss.to_csv(csv_name, index=False)
        print(f"\n>>> [表格已保存] SDE loss_dps 数据已存入: {csv_name}")
        
    except Exception as e:
        print(f"\n>>> [保存失败]: {e}")
    # =====================================================================
    '''
    return x_to_sample(x_mean).detach().cpu().numpy(), x_generated if save_sample_path else None, losses


def generate_parallel_1d(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, gamma=1.e-2,
                          num_steps=10, overlap=1,
                              device='cpu', dtype='float32', eps=1e-3, save_sample_path=False,
                              probability_flow=False, continuous=True, data_scalar=None):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,)
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    # x_known = torch.from_numpy(x0).to(device).type(dtype_torch)
    y = torch.from_numpy(y).to(device).type(dtype_torch)
    # shape_sample = [len(y), config.num_channels, config.image_size, config.image_size]

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(y), b, nf, ncomp+config.num_conditions, config.image_size]       # batch*b*nf*(c+npara)*h
    # shape_sample = [config.num_samples, ns_real, ncomp+config.num_modals-1, config.image_size]     # batch*ns_real*(c+npara)*h

    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp+config.num_conditions, config.image_size], dtype=dtype_torch, device=device)   # batch*ns_real*(c+npara)*h
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    x_unknown = sde.prior_sampling(shape).to(device).float()    # batch*b*nf*(c+npara)*h
    
    x = x_unknown           # batch*b*(nf*c+npara)*h*w
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[])
    with tqdm(range(sde.N)) as tqdm_setting:
        for i in range(sde.N):
            t = timesteps[i]

            '''method 1 (batched)'''
            vec_t = torch.ones(shape[0]*b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h -> (b n) t c h')       # (batch*b)*nf*(c+npara)*h*w

            '''corrector'''
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)     # (batch*b)*nf*(c+npara)*h*w

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h -> (b n) t c h')

            with torch.enable_grad():
                inp = temp.clone()                  # (batch*b)*nf*(c+npara)*h*w
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)          # (batch*b)*nf*(c+npara)*h*w
                with torch.no_grad():
                    f, G = sde.discretize(temp, vec_t)
                    rev_f = f - G[:, None, None, None] ** 2 * score.detach() * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    temp_mean = temp - rev_f
                    temp_u = temp_mean + rev_G[:, None, None, None] * zb
                # dps loss
                _, std = sde.marginal_prob(xb, vec_t)
                if isinstance(sde, VESDE):
                    x0_hat = rearrange(std[:, None, None, None] ** 2 * score + inp, '(b n) t c h -> b n t c h', n=b)     # batch*b*nf*(c+npara)*h
                else:
                    alpha_sqrt_ = (1-std**2).sqrt()[:, None, None, None]
                    x0_hat = rearrange((std[:, None, None, None] ** 2 * score + inp)/alpha_sqrt_, '(b n) t c h -> b n t c h', n=b)     # batch*b*nf*(c+npara)*h
                x0_hat_temp = x_to_sample(x0_hat)
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())

                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_temp)) ** 2 / var).reshape(x0_hat.shape[0], -1), dim=-1)  # /scalar.sqrt()  *std[None, :, None, None]
                loss_dps = torch.sum(loss_dps, dim=0)  # /loss_dps.detach().mean().sqrt()
                if std_y is not None:
                    loss_dps = loss_dps/2.
                # loss_dps = loss_dps/loss_dps.detach().sqrt()    # normalize

                loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)    # , len(x0_hat[0])    /std[:, None, None].sqrt()
                loss_consis = torch.sum(loss_consis)        # *torch.softmax(loss_consis.detach(), dim=1)
                # loss_consis = loss_consis/loss_consis.detach().sqrt()    # normalize
                loss_consis_para = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, ncomp:]-x0_hat[:, 1:, :ol, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_consis_para = torch.sum(loss_consis_para, dim=0)
                # loss_consis_para = loss_consis_para/loss_consis_para.detach().sqrt()    # normalize

                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e}')
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                temp = temp_u - dx     # (batch*b)*(nf*c+npara)*h*w
            #     # x = x_u
            temp = temp.detach()

            x = rearrange(temp, '(b n) t c h -> b n t c h', n=b)
            x_mean = rearrange(temp_mean, '(b n) t c h -> b n t c h', n=b)
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            tqdm_setting.update(1)

    return x_to_sample(x_mean).detach().cpu().numpy(), x0_hats if save_sample_path else None, losses


def vor_cal(u, v, grid_num, x_range):
    dx = (x_range[1]-x_range[0])/grid_num
    vor = (v[:-1, 1:]-v[:-1, :-1])/dx-(u[1:, :-1]-u[:-1, :-1])/dx
    return vor


def vor_cal_batch(x, grid_num, x_range, reverse=False, method='diff_1st', is_stagger=True):
    # method: 'diff_1st', 'spectral'
    vor = []
    for v in x:
        vx, vy = (v[1], v[0]) if reverse else (v[0], v[1])
        if 'diff_1st' in method:
            vor.append(vor_cal(vx, vy, grid_num, x_range))
        elif 'spectral' in method:
            vor.append(vor_cal_spectral(vx, vy, is_stagger=is_stagger))
        else:
            raise NotImplementedError('No such method for vorticity calculation!')
    return np.array(vor)


def vor_cal_plus(u, v, grid_num, x_range):
    omega = np.zeros((grid_num, grid_num))
    dx = dy = (x_range[1] - x_range[0]) / grid_num
    for i in range(1, grid_num - 1):
        for j in range(1, grid_num - 1):
            dudy = (u[i, j + 1] - u[i, j - 1]) / (2 * dy)
            dvdx = (v[i + 1, j] - v[i - 1, j]) / (2 * dx)
            omega[i, j] = dvdx - dudy
    return omega


def vor_cal_spectral(u, v, is_stagger=True):
    if is_stagger:
        # for staggered grid arrangement, we interpolate velocities from cell faces to cell centres
        u = 0.5 * (u + np.roll(u, 1, axis=1))
        v = 0.5 * (v + np.roll(v, -1, axis=0))
    k_max = len(u)//2
    k = np.concatenate([np.arange(0, k_max, 1), np.arange(-k_max, 0, 1)])
    k_x, k_y = np.meshgrid(k, k)
    F_u = np.fft.fft2(u)
    F_v = np.fft.fft2(v)
    # F_ux = 1j * k_x * F_u
    F_uy = 1j * k_y * F_u
    F_vx = 1j * k_x * F_v
    # F_vy = 1j * k_y * F_v
    # ux = np.fft.ifft2(F_ux)
    uy = np.fft.irfft2(F_uy[..., :k_max+1])
    vx = np.fft.irfft2(F_vx[..., :k_max+1])
    # vy = np.fft.ifft2(F_vy)
    return vx - uy


def mask_gen(input_shape, mask_ratio=0.5, seed=None):
    m = np.ones(input_shape)

    indices = [np.arange(i) for i in input_shape]
    I = np.meshgrid(*indices, indexing='ij')
    indices = np.array([index.reshape(-1) for index in I]).transpose(1, 0)
    num_pixel = len(indices)
    if seed is None:
        i_indices = np.random.choice(num_pixel, int(mask_ratio*num_pixel), replace=False)
    else:
        rng = np.random.RandomState(seed)
        i_indices = rng.choice(num_pixel, int(mask_ratio * num_pixel), replace=False)
    indices = indices[i_indices]
    m[tuple(indices.transpose(1, 0))] = 0
    m = m.astype(bool)
    return m


def plot_field(fields, row, col, dpi=100, q_range=None, save_fig=None):
    figsize = (col, row)
    fig, axes = plt.subplots(row, col, tight_layout=True, figsize=figsize, dpi=dpi)
    fields = fields.reshape(row, col, *fields.shape[1:])
    for i in range(row):
        for j in range(col):
            field = fields[i, j]
            pc = axes[i, j].pcolormesh(field, cmap='RdBu_r')
            if q_range is not None:
                pc.set_clim(q_range)
            axes[i, j].axis('off')
            axes[i, j].set_aspect(1)
    plt.show()
    if save_fig is not None:
        fig.savefig('./results/'+save_fig)


def cal_water_attr(t):
    attr_table = dict(
        t=[10, 20, 30, 40],
        rho=[999.7, 998.2, 995.7, 992.2],
        lamb=[0.574, 0.599, 0.618, 0.635],
        cp=[4191., 4183., 4174., 4174],
        # alpha=[20.e-6, 21.4e-6, 22.9e-6, 24.3e-6],
        mu=[1.306e-3, 1.004e-3, 0.8015e-4, 0.6533e-4],
        nu=[1.306e-6, 1.006e-6, 0.8050e-6, 0.6590e-6],
        Pr=[9.52, 7.02, 5.42, 4.31],
                      )
    t_low, t_high = attr_table['t'][0], attr_table['t'][-1]
    if t < t_low or t > t_high:
        raise ValueError(f'Input temperature out of range! Expect the input in range {t_low} to {t_high}')
    xs = attr_table['t']
    attr_t = dict()
    for key in attr_table.keys():
        v = attr_table[key]
        f = interp1d(xs, v, kind='linear')
        attr_t[key] = f(t)
    return attr_t


def sample_to_hot_wire(sample, coords, spacing, offsets=np.array([0, 0]), num_frame=10,
                       scalar=None, is_avg=True, use_para=False, weight=1.):
    # sample: b*(t*c+2)*h*w; coords: N*dim, N-points measurements of velocity; spacing: grid spacing
    device = sample.device
    indices = ((coords+offsets)/spacing).astype('int')

    if len(sample.shape) > 4:
        para = sample[:, :, num_frame:]
        sample = sample[:, :, :num_frame]
    else:
        para = sample[:, num_frame:num_frame+1]
        sample = sample[:, :num_frame]

    if scalar is not None:
        # scalar_std = torch.ones([1, len(sample[0]), 1, 1]).to(device)
        # scalar_mean = torch.zeros([1, len(sample[0]), 1, 1]).to(device)
        # scalar_std[:, :num_frame] = scalar_std[:, :num_frame]*scalar.std
        # scalar_mean[:, :num_frame] = scalar_mean[:, :num_frame]+scalar.mean
        # sample = sample*scalar.std+scalar.mean
        sample = scalar(sample)

    if len(sample.shape) > 4:
        obs = sample[:, :, :, indices[:, 0], indices[:, 1]]
        obs = torch.sqrt(obs[:, :, 0]**2+obs[:, :, 1]**2)
    else:
        obs = sample[:, :, indices[:, 0], indices[:, 1]]
        obs = torch.sqrt(obs[:, ::2]**2+obs[:, 1::2]**2)
    if is_avg:
        obs = obs.mean(1)
    if use_para:
        # obs = torch.cat([obs.reshape(len(obs), -1), weight*para.reshape(len(para), -1).mean(1)[:, None]], dim=-1)
        obs = torch.cat([obs.reshape(len(obs), -1), weight*para.reshape(len(para), -1)], dim=-1)

    return obs


def cal_rmse(gt, pred, normalize=True, reduct='sum'):
    # reduct = 'sum' or 'mean' etc.
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    rmse = []
    for a, b in zip(gt, pred):
        if normalize:
            coeff = 1./lib_name.sqrt(reduct_fn(a**2))
        else:
            coeff = 1.
        rmse.append(coeff*lib_name.sqrt(reduct_fn((a-b)**2)))
    return np.array(rmse) if isinstance(a, np.ndarray) else rmse


def cal_correlation(gt, pred, standardize=True, reduct='sum'):
    # standardize: whether to substract mean value of input data
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    cossim = []
    for a, b in zip(gt, pred):
        if standardize:
            a_mean = lib_name.mean(a)
            b_mean = lib_name.mean(b)
        else:
            a_mean = 0.
            b_mean = 0.
        a_norm = lib_name.sqrt(reduct_fn(a**2))
        b_norm = lib_name.sqrt(reduct_fn(b**2))
        cossim.append(reduct_fn((a-a_mean).reshape(-1)*(b-b_mean).reshape(-1))/(a_norm*b_norm))
    return np.array(cossim) if isinstance(a, np.ndarray) else cossim


def voronoi_interp(matrix, mask):
    """
    Completes a masked matrix using voronoi-tessellation interpolation, compatible with
    both numpy.ndarray and torch.Tensor.

    Args:
        matrix (ndarray or Tensor): 2D array containing the masked matrix.
        mask (ndarray or Tensor): 2D bool-type array or tensor.

    Returns:
        completed_matrix (ndarray or Tensor): 2D array or tensor containing the completed matrix.
    """
    
    is_tensor = torch.is_tensor(matrix)

    # Convert to NumPy if input is Tensor
    if is_tensor:
        device = matrix.device
        matrix_np = matrix.detach().cpu().numpy()
        mask_np = mask.numpy()
    else:
        matrix_np = matrix
        mask_np = mask

    # Find the indices of the masked points
    unmasked_indices = np.argwhere(mask_np == True)
    vor = Voronoi(unmasked_indices)
    values = matrix_np[unmasked_indices[:, 0], unmasked_indices[:, 1]]

    # Loop over each masked point and fill it in using the nearest-neighbor value
    grid_x, grid_y = np.meshgrid(range(matrix_np.shape[0]), range(matrix_np.shape[1]), indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    tree = cKDTree(vor.points)
    _, indexes = tree.query(grid_points)
    voronoi_matrix = values[indexes].reshape(grid_x.shape)

    # Convert back to Tensor if the input was Tensor
    if is_tensor:
        return torch.from_numpy(voronoi_matrix).to(device)
    else:
        return voronoi_matrix


def Kuramoto_Sivashinsky_equation(N, h, tmax, vis, init_frame, nplt=1, M=16):
    # Initial condition and grid setup
    x = np.transpose(np.conj(np.arange(1, N+1))) / N
    a = -1
    b = 1
    # Generate samples from the Gaussian process
    # sample = gaussian_process_periodic(np.linspace(0, 1, N), init_sigma, init_amp)
    u = init_frame     # np.cos(x/16)*(1+np.sin(x/16))
    v = np.fft.fft(u)
    # scalars for ETDRK4
    k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) / 16
    L = k**2 - vis*k**4
    E = np.exp(h*L)
    E_2 = np.exp(h*L/2)
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    # main loop
    uu = np.array([u])
    tt = 0
    nmax = round(tmax/h)
    g = -0.5j*k
    for n in range(1, nmax+1):
        t = n*h
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E_2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E_2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        if n%nplt == 0:
            u = np.real(np.fft.ifft(v))
            uu = np.append(uu, np.array([u]), axis=0)
            tt = np.hstack((tt, t))
    return uu


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def s3gm_sample_2d_vesde_ddim(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=1000,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True):
    """
    Performs 2D inpainting and generation using a combination of parallel and
    autoregressive sampling, utilizing generate_parallel_2d_vesde_ddim.
    """
    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_2d_vesde_ddim(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                                    probability_flow=probability_flow, continuous=continuous,
                                                    denoising_steps=denoising_steps,eta=eta)

    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = generate_ar_2d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                    n_steps=n_steps, probability_flow=probability_flow,
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        # Concatenate the generated extra part, excluding the overlapping portion from x_extra
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y


def generate_parallel_2d_vesde_ddim(config, net, sde, y, transform, corrector, n_steps=5, 
                          alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                          gamma=1.e-2, num_steps=None,  
                          denoising_steps=100, 
                          overlap=1, eta=0.0,  
                          device='cpu', dtype='float32', eps=1e-3, 
                          save_sample_path=False, probability_flow=False, 
                          continuous=True, data_scalar=None):
    
    dtype_torch = getattr(torch, dtype)
    
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
                                           corrector=corrector,
                                           continuous=continuous,
                                           snr=snr,
                                           n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))  # 需要生成的样本数量
    ns_real = b * (nf - ol) + ol  # 实际生成的步数
    nc = config.num_conditions
    ds = denoising_steps
    #print(ds)
    #print(eta)
    # 样本形状定义
    shape = [len(y), b, nf, ncomp + nc, config.image_size, config.image_size]
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size, config.image_size], 
                                dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    # 创建DDIM子序列时间步 - 使用denoising_steps控制迭代步数
    timesteps = torch.linspace(sde.T, eps, ds + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    #print(timesteps)
    # 初始化：从先验分布采样
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        """计算总变分正则化损失"""
        # 时间方向差分
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        # 空间方向差分
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
    
        # 组合正则化项
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        return reg_loss
    # 动量缓存
    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), x.size(5), device=x.device)
    
    # 损失跟踪
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[])
    
    # 获取sigma序列函数
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    # 使用denoising_steps作为迭代循环次数
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            # 将样本重组为批量
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')
            
            # 校正步骤（可选）
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)
            
            # DDIM采样核心计算
            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                
                # 计算得分函数
                score = net_fn(inp, vec_t)
                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    # 添加必要的维度用于广播
                    sigma_current = sigma_current[:, None, None, None, None]
                
                # 计算x0的估计值
                x0_hat = inp + (sigma_current ** 2) * score
                
                ################################################
                # 关键修复：精确处理时间步参数
                ################################################
                
                # 创建用于计算下一个时间步sigma的适当形状输入
                dummy_shape = torch.Size([inp.shape[0]]) + torch.Size([1, 1, 1, 1])
                dummy_input = torch.zeros(dummy_shape, device=inp.device, dtype=inp.dtype)
                
                # 创建与当前输入形状匹配的时间向量
                vec_t_next = t_next * torch.ones_like(vec_t)
                
                # 获取下一个时间步的sigma
                _, sigma_next = sde.marginal_prob(dummy_input, vec_t_next)
                if sigma_next.dim() == 1:
                    # 添加必要的维度用于广播
                    sigma_next = sigma_next[:, None, None, None, None]
                
                # 计算比例因子
                ratio = sigma_next / sigma_current
                
                # 计算确定性部分
                # 确保所有操作数具有相同维度
                x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                
                ################################################
                # 添加可控噪声
                ################################################
                
                noise = torch.randn_like(inp)
                
                # 计算方差项 - 根据DDIM公式
                variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                
                # 确保方差非负
                variance = torch.clamp(variance, min=0)
                
                # 添加噪声项
                noise_coeff = torch.sqrt(variance)
                x_pred = x_pred_mean + noise_coeff * noise
                
                # 损失计算
                x0_hat_reshaped = rearrange(x0_hat, '(b n) t c h w -> b n t c h w', n=b)
                x0_hat_sample = x_to_sample(x0_hat_reshaped)
                
                #print(f"x0_hat_sample 形状: {x0_hat_sample.shape}")
                #print(f"y 形状: {y.shape}")
                
                
                # 观测损失
                var = std_y**2 + gamma * sigma_current**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_sample))**2 / var).reshape(x0_hat_reshaped.shape[0], -1), dim=-1)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                # 一致性损失
                loss_consis = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, :ncomp].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, :ncomp])**2
                )
                loss_consis = torch.sum(loss_consis, dim=-1).mean()
                
                # 参数一致性损失
                loss_consis_para = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, ncomp:].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, ncomp:])**2
                )
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                # 计算正则化损失
                reg_loss = compute_regularization(x0_hat_sample, 1e-1)
                
                # 总损失
                #loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para + 100000 * reg_loss
                
                # 损失记录
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e}')
                
                # 梯度调整
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                momentum = 0.4 * momentum + 0.6 * dx
                x_pred = x_pred - momentum
                #x_pred = x_pred - dx
            
            # 重组样本进入下一步
            x = rearrange(x_pred, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(x_pred_mean, '(b n) t c h w -> b n t c h w', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses

def s3gm_sample_2d_vesde_ddim_ad(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=1000,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True):
    """
    Performs 2D inpainting and generation using a combination of parallel and
    autoregressive sampling, utilizing generate_parallel_2d_vesde_ddim.
    """
    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_2d_vesde_ddim_ad(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                                    probability_flow=probability_flow, continuous=continuous,
                                                    denoising_steps=denoising_steps,eta=eta)
    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = generate_ar_2d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                    n_steps=n_steps, probability_flow=probability_flow,
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        # Concatenate the generated extra part, excluding the overlapping portion from x_extra
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y

def generate_parallel_2d_vesde_ddim_ad(config, net, sde, y, transform, corrector, n_steps=5, 
                          alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                          gamma=1.e-2, num_steps=None,  
                          denoising_steps=100, 
                          overlap=1, eta=0.0,  
                          device='cpu', dtype='float32', eps=1e-3, 
                          save_sample_path=False, probability_flow=False, 
                          continuous=True, data_scalar=None):
    
    dtype_torch = getattr(torch, dtype)
    
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
                                           corrector=corrector,
                                           continuous=continuous,
                                           snr=snr,
                                           n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))  # 需要生成的样本数量
    ns_real = b * (nf - ol) + ol  # 实际生成的步数
    nc = config.num_conditions
    ds = denoising_steps
    #print(ds)
    #print(eta)
    # 样本形状定义
    shape = [len(y), b, nf, ncomp + nc, config.image_size, config.image_size]
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size, config.image_size], 
                                dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    # ===================== 修改开始 =====================
    # 创建自适应时间步序列 - 在高方差区稀疏，低方差区密集
    # 提取出sde.T和sde.eps作为常量，避免函数传递
    T_max = sde.T
    eps_min = eps  # 使用函数参数中的eps
    
    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        """
        修正的时间步函数 - 确保所有时间步在[T, eps]范围内
        """
        # 1. 生成基础序列
        s = torch.linspace(0, 1, steps + 1, device=device)
    
        # 2. 应用余弦函数
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
    
        # 3. 直接映射到时间范围 [T, eps]
        # 因为alpha_bar从1递减到0，我们直接映射到[T, eps]
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
    
        # 4. 确保严格递减
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
    
        # 5. 精确设置边界值
        adaptive_t[0] = T
        adaptive_t[-1] = eps
    
        return adaptive_t
    
    # 使用更稳健的自适应时间步
    timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    eps_t = torch.tensor(eps_min, device=device).float()
    # ===================== 修改结束 =====================
    #print(timesteps)
    # 初始化：从先验分布采样
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []


    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        """计算总变分正则化损失"""
        # 时间方向差分
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        # 空间方向差分
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
    
        # 组合正则化项
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        return reg_loss
    # 动量缓存
    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), x.size(5), device=x.device)
    
    # 损失跟踪
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[])
    
    # 获取sigma序列函数
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    # 使用denoising_steps作为迭代循环次数
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            # 将样本重组为批量
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')
            
            # 校正步骤（可选）
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)
            
            # DDIM采样核心计算
            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                
                # 计算得分函数
                score = net_fn(inp, vec_t)
                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    # 添加必要的维度用于广播
                    sigma_current = sigma_current[:, None, None, None, None]
                
                # 计算x0的估计值
                x0_hat = inp + (sigma_current ** 2) * score
                
                ################################################
                # 关键修复：精确处理时间步参数
                ################################################
                
                # 创建用于计算下一个时间步sigma的适当形状输入
                dummy_shape = torch.Size([inp.shape[0]]) + torch.Size([1, 1, 1, 1])
                dummy_input = torch.zeros(dummy_shape, device=inp.device, dtype=inp.dtype)
                
                # 创建与当前输入形状匹配的时间向量
                vec_t_next = t_next * torch.ones_like(vec_t)
                
                # 获取下一个时间步的sigma
                _, sigma_next = sde.marginal_prob(dummy_input, vec_t_next)
                if sigma_next.dim() == 1:
                    # 添加必要的维度用于广播
                    sigma_next = sigma_next[:, None, None, None, None]
                
                # 计算比例因子
                ratio = sigma_next / sigma_current
                
                # 计算确定性部分
                # 确保所有操作数具有相同维度
                x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                
                ################################################
                # 添加可控噪声
                ################################################
                
                noise = torch.randn_like(inp)
                
                # 计算方差项 - 根据DDIM公式
                variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                
                # 确保方差非负
                variance = torch.clamp(variance, min=0)
                
                # 添加噪声项
                noise_coeff = torch.sqrt(variance)
                x_pred = x_pred_mean + noise_coeff * noise
                
                # 损失计算
                x0_hat_reshaped = rearrange(x0_hat, '(b n) t c h w -> b n t c h w', n=b)
                x0_hat_sample = x_to_sample(x0_hat_reshaped)
                
                #print(f"x0_hat_sample 形状: {x0_hat_sample.shape}")
                #print(f"y 形状: {y.shape}")
                
                
                # 观测损失
                var = std_y**2 + gamma * sigma_current**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_sample))**2 / var).reshape(x0_hat_reshaped.shape[0], -1), dim=-1)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                # 一致性损失
                loss_consis = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, :ncomp].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, :ncomp])**2
                )
                loss_consis = torch.sum(loss_consis, dim=-1).mean()
                
                # 参数一致性损失
                loss_consis_para = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, ncomp:].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, ncomp:])**2
                )
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                # 计算正则化损失
                reg_loss = compute_regularization(x0_hat_sample, 1e-1)
                
                # 总损失
                #loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para + 100000 * reg_loss
                
                # 损失记录
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e}')
                
                # 梯度调整
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8) 

                momentum = 0.4 * momentum + 0.6 * dx
                x_pred = x_pred - momentum
                #x_pred = x_pred - dx
            
            # 重组样本进入下一步
            x = rearrange(x_pred, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(x_pred_mean, '(b n) t c h w -> b n t c h w', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses

def s3gm_sample_2d_vesde_ddim_ad_mix(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=1000,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True):
    """
    Performs 2D inpainting and generation using a combination of parallel and
    autoregressive sampling, utilizing generate_parallel_2d_vesde_ddim.
    """
    if T_prime_y > 0:

        x_y, _, _ = generate_parallel_2d_vesde_ddim_ad_mix(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                                    probability_flow=probability_flow, continuous=continuous,
                                                    denoising_steps=denoising_steps,eta=eta)
        '''
        x_y, _, _ = generate_parallel_2d(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                    probability_flow=probability_flow, continuous=continuous)
        '''
    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]

        x_extra, _ = generate_ar_2d_ISS(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                    n_steps=n_steps, probability_flow=probability_flow,
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        '''
        x_extra, _ = generate_ar_2d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                    n_steps=n_steps, probability_flow=probability_flow,
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        '''
        # Concatenate the generated extra part, excluding the overlapping portion from x_extra
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y

def generate_parallel_2d_vesde_ddim_ad_mix(config, net, sde, y, transform, corrector, n_steps=5, 
                          alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                          gamma=1.e-2, num_steps=None,  
                          denoising_steps=100, 
                          overlap=1, eta=0.0,  
                          device='cpu', dtype='float32', eps=1e-3, 
                          save_sample_path=False, probability_flow=False, 
                          continuous=True, data_scalar=None):
    
    dtype_torch = getattr(torch, dtype)
    
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
                                           corrector=corrector,
                                           continuous=continuous,
                                           snr=snr,
                                           n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))
    ns_real = b * (nf - ol) + ol 
    nc = config.num_conditions
    ds = denoising_steps
    
    shape = [len(y), b, nf, ncomp + nc, config.image_size, config.image_size]
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size, config.image_size], 
                                dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    T_max = sde.T
    eps_min = eps
    
    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    #timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    
    # 初始化：从先验分布采样
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), x.size(5), device=x.device)
    
    # 损失跟踪
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[])
    
    # 获取sigma序列函数
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    # 计算切换步数阈值
    sde_start_step = ds - 10
    
    # 使用denoising_steps作为迭代循环次数
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next  # 计算时间步长
            
            # 将样本重组为批量
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')
            
            # 校正步骤（可选）
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)
            
            # ISS/SDE采样核心计算
            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                
                # 计算分数函数
                score = net_fn(inp, vec_t)
                
                #print(inp.shape)
                #print(vec_t.shape)
                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]
                
                # 计算x0的估计值
                x0_hat = inp + (sigma_current ** 2) * score
                
                # 混合采样策略：前70%使用ISS，后30%使用SDE
                if i < sde_start_step: 
                    # ======== DDIM 采样部分 ========
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    # 添加可控噪声
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    x_pred = x_pred_mean + noise_coeff * noise

                    
                else:
                    # ======== SDE采样部分 ========
                    with torch.no_grad():
                        # 使用sde.discretize替代sde.sde
                        f, G_val = sde.discretize(inp, vec_t)
        
                        # rev_f = f - G**2 * s_theta
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0
                        #print(rev_f)
                        #print(G_val)
                        # 计算确定性部分
                        temp_mean = inp - rev_f
        
                        # 添加噪声项
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                
                # 损失计算
                x0_hat_reshaped = rearrange(x0_hat, '(b n) t c h w -> b n t c h w', n=b)
                x0_hat_sample = x_to_sample(x0_hat_reshaped)
                
                # 观测损失
                var = std_y**2 + gamma * sigma_current**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_sample))**2 / var).reshape(x0_hat_reshaped.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                # 一致性损失
                loss_consis = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, :ncomp].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, :ncomp])**2
                )
                loss_consis = torch.sum(loss_consis, dim=-1).mean()
                
                # 参数一致性损失
                loss_consis_para = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, ncomp:].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, ncomp:])**2
                )
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                # 计算正则化损失
                #decay_factor = max(0.1, 1.0 - (i / ds)) 
                reg_loss = 1 * compute_regularization(x0_hat_sample, 1e4)
                
                # 总损失
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para +  reg_loss
                #loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                
                # 损失记录
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                losses['reg_loss'].append(reg_loss.item())
                
                # 显示当前采样模式
                mode = "ISS" if i < sde_start_step else "SDE"
                tqdm_setting.set_description(f'{mode} | loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e} | reg_loss.: { reg_loss.item():.5e}')
                
                # 梯度调整
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8) 
                momentum = 0.01 * momentum + 0.99 * dx
                #omentum = 0.99 * momentum + 0.01 * dx
                x_pred = x_pred - momentum
                #x_pred = x_pred - dx
            
            # 重组样本进入下一步
            x_pred = x_pred.detach()
            #print(x_pred.shape)
            x = rearrange(x_pred, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(x_pred if i < sde_start_step else temp_mean,  # SDE没有mean，直接使用x_pred
                              '(b n) t c h w -> b n t c h w', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses

def generate_ar_2d_ISS(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1., mult=10, transform_init=None, num_steps=10, overlap=1,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            )
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    ds = sde.N
    
    T_max = sde.T
    eps_min = eps
    
    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    #timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    sde_start_step = ds - 10
    #sde_start_step = 0
    
    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(x0), nf, ncomp+config.num_conditions, config.image_size, config.image_size]       # batch*nf*(c+npara)*h*w

    transform_ar, alpha_ar = lambda x: x[:, :ol], alpha
    transform_init, alpha_init = transform_ar if transform_init is None else transform_init, alpha*mult

    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
            transform = transform_init
            alpha = alpha_init
        else:
            y = x_mean[:, -ol:].detach()
            transform = transform_ar
            alpha = alpha_ar
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(ds)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next  
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)

                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]

                x0_hat = inp + (sigma_current ** 2) * score
            
                if i < sde_start_step:
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    x_u = x_pred_mean
                    
                else:
                    with torch.no_grad():
                        '''
                        f, G_val = sde.discretize(inp, vec_t)
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0
                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                        x_u = x_pred
                        '''
                        f, G = sde.discretize(x, vec_t)
                        rev_f = f - G[:, None, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                        rev_G = torch.zeros_like(G) if probability_flow else G
                        x_mean = x - rev_f
                        x_u = x_mean + rev_G[:, None, None, None, None] * z
                
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                #x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                loss = alpha * loss_dps
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated




def s3gm_sample_1d_ISS(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                T_prime_y=10, T_prime=0, overlap=1,
                              denoising_steps=100, eta=0,
                              device='cpu', dtype='float32', eps=1e-12, 
                              probability_flow=False, continuous=True,reg_coef_pa=1e4):
    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_1d_ISS(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, denoising_steps=denoising_steps, overlap=overlap, eta=eta,
                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                    probability_flow=probability_flow, continuous=continuous,reg_coef_pa=reg_coef_pa)
    else:
        x_y = y
    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = generate_ar_1d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0, 
                                    n_steps=n_steps, probability_flow=probability_flow, 
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
    return np.concatenate([x_y, x_extra[:, overlap:]], axis=1) if T_prime > x_y.shape[1] else x_y

def generate_parallel_1d_ISS(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, gamma=1.e-2,
                          num_steps=10, denoising_steps=100,overlap=1, eta=0.0,
                              device='cpu', dtype='float32', eps=1e-3, save_sample_path=False,
                              probability_flow=False, continuous=True, data_scalar=None,reg_coef_pa=1e4):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,)
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    # x_known = torch.from_numpy(x0).to(device).type(dtype_torch)
    y = torch.from_numpy(y).to(device).type(dtype_torch)
    # shape_sample = [len(y), config.num_channels, config.image_size, config.image_size]

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(y), b, nf, ncomp+config.num_conditions, config.image_size]       # batch*b*nf*(c+npara)*h
    ds = denoising_steps
    # shape_sample = [config.num_samples, ns_real, ncomp+config.num_modals-1, config.image_size]     # batch*ns_real*(c+npara)*h

    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp+config.num_conditions, config.image_size], dtype=dtype_torch, device=device)   # batch*ns_real*(c+npara)*h
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample
    
    T_max = sde.T
    
    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    #timesteps = adaptive_cosine_timesteps(ds, T_max, eps, 0.008)

    timesteps = torch.linspace(T_max, eps, ds + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    x_unknown = sde.prior_sampling(shape).to(device).float()    # batch*b*nf*(c+npara)*h
    
    x = x_unknown           # batch*b*(nf*c+npara)*h*w
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:] - x0_hat_temp[:, :, :, :-1])
        reg_loss = lambda_reg * (10*dx_t.mean() + dx_w.mean())
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), device=x.device)
    
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[])
    
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    #sde_start_step = int(ds * 0.999)  # 999%后开始SDE采样
    sde_start_step = ds - 10

    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next
            
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h -> (b n) t c h')
            
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h -> (b n) t c h')

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                
                score = net_fn(inp, vec_t)
                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    temp_u = x_pred_mean + noise_coeff * noise

                else:
                    with torch.no_grad():
                        f, G = sde.discretize(inp, vec_t)
        
                        # rev_f = f - G**2 * s_theta
                        rev_f = f - G[:, None, None, None] ** 2 * score.detach() * 1.0
                        rev_G = torch.zeros_like(G) if probability_flow else G
                        temp_mean = inp - rev_f 
                        temp_u = temp_mean + rev_G[:, None, None, None] * zb
                
                _, std = sde.marginal_prob(xb, vec_t)
                x0_hat = rearrange(std[:, None, None, None] ** 2 * score + inp, '(b n) t c h -> b n t c h', n=b)
                x0_hat_temp = x_to_sample(x0_hat)
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())

                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_temp)) ** 2 / var).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)  # /loss_dps.detach().mean().sqrt()
                if std_y is not None:
                    loss_dps = loss_dps/2.

                loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)    # , len(x0_hat[0])    /std[:, None, None].sqrt()
                #loss_consis = torch.sum(loss_consis)        # *torch.softmax(loss_consis.detach(), dim=1)
                loss_consis = torch.sum(loss_consis, dim=-1).mean()

                loss_consis_para = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, ncomp:]-x0_hat[:, 1:, :ol, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                #loss_consis_para = torch.sum(loss_consis_para, dim=0)
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                #decay_factor = max(0.1, 1.0 - (i / ds)) 
                reg_loss = 1 * compute_regularization(x0_hat_temp, reg_coef_pa)
                
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para +  reg_loss
                #loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                losses['reg_loss'].append(reg_loss.item())
                
                mode = "ISS" if i < sde_start_step else "SDE"
                tqdm_setting.set_description(f'{mode} | loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e} | reg_loss.: { reg_loss.item():.5e}')
                
                # 梯度调整
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8) 
                #momentum = 0.01 * momentum + 0.99 * dx
                #x_pred = x_pred - momentum
                temp = temp_u - dx
                #temp = temp_u - momentum
            
            # 重组样本进入下一步
            temp = temp.detach()
            #print(x_pred.shape)
            x = rearrange(temp, '(b n) t c h -> b n t c h', n=b)
            x_mean = rearrange(temp_u if i < sde_start_step else temp_mean,  # SDE没有mean，直接使用x_pred
                              '(b n) t c h -> b n t c h', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), (x_generated if save_sample_path else None), losses












def compute_acoustic_residual_torch(p, dx, dt, c, source_term=None):
    """
    PyTorch版本的声学波动方程残差计算（支持自动微分）
    """
    # 确保输入张量 p 是 3维的，且形状为 (1, T, N)
    assert p.ndim == 3, f"Expected p to be 3D (1, T, N), but got shape {p.shape}"
    assert p.shape[0] == 1, f"Expected first dim to be 1 (batch), but got {p.shape[0]}"
    
    # 去掉多余的 batch 维度（dim=0，值为1），变成 (T, N)
    p_2d = p[0, :, :]  # shape: (T, N)
    
    # 确保 p_2d 仍然需要梯度
    # print(p_2d.requires_grad) # 您可以在这里添加一个检查点

    T, N = p_2d.shape
    
    # 只计算内部点
    d2p_dt2 = (p_2d[2:, 1:-1] - 2 * p_2d[1:-1, 1:-1] + p_2d[:-2, 1:-1]) / (dt ** 2)
    d2p_dx2 = (p_2d[1:-1, 2:] - 2 * p_2d[1:-1, 1:-1] + p_2d[1:-1, :-2]) / (dx ** 2)
    
    # 波动方程残差
    residual = (1 / c ** 2) * d2p_dt2 - d2p_dx2
    
    if source_term is not None:
        assert source_term.ndim == 3, f"source_term 应为 3D (1, T, N)，但得到 {source_term.shape}"
        assert source_term.shape[0] == 1, f"source_term 的 batch 维度应为 1，但得到 {source_term.shape[0]}"
        source_term_2d = source_term[0, :, :]
        source_inner = source_term_2d[1:-1, 1:-1]
        residual = residual - source_inner
    
    return residual



def s3gm_sample_1d_pde(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                T_prime_y=10, T_prime=0, overlap=1,
                              device='cpu', dtype='float32', eps=1e-12, 
                              probability_flow=False, continuous=True,
                              residual_n_steps = 20, alpha_residual = 1e-3,m_steps=10,lr=0.05):
    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_1d_pde(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                    probability_flow=probability_flow, continuous=continuous,
                                    residual_n_steps = residual_n_steps, alpha_residual = alpha_residual,m_steps=m_steps,lr=lr)
    else:
        x_y = y
    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = generate_ar_1d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0, 
                                    n_steps=n_steps, probability_flow=probability_flow, 
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
    return np.concatenate([x_y, x_extra[:, overlap:]], axis=1) if T_prime > x_y.shape[1] else x_y



def generate_parallel_1d_pde(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, gamma=1.e-2,
                          num_steps=10, overlap=1,
                              device='cpu', dtype='float32', eps=1e-3, save_sample_path=False,
                              probability_flow=False, continuous=True, data_scalar=None,
                              residual_n_steps = 20, alpha_residual = 3e-3,m_steps=10,lr=0.05):
    dtype_torch = getattr(torch, dtype)
   
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    residual_n_steps = residual_n_steps
    alpha_residual = alpha_residual   # 残差损失的权重系数（需调优）
    source_term = None     # 若有已知声源项，传入对应数据
    m_steps=m_steps
    lr=lr
    
    # x_known = torch.from_numpy(x0).to(device).type(dtype_torch)
    y = torch.from_numpy(y).to(device).type(dtype_torch)
    # shape_sample = [len(y), config.num_channels, config.image_size, config.image_size]

    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(y), b, nf, ncomp+config.num_conditions, config.image_size]       # batch*b*nf*(c+npara)*h
    # shape_sample = [config.num_samples, ns_real, ncomp+config.num_modals-1, config.image_size]     # batch*ns_real*(c+npara)*h

    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp+config.num_conditions, config.image_size], dtype=dtype_torch, device=device)   # batch*ns_real*(c+npara)*h
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps, device=device).float()
    x_unknown = sde.prior_sampling(shape).to(device).float()    # batch*b*nf*(c+npara)*h
    
    x = x_unknown           # batch*b*(nf*c+npara)*h*w
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], loss_residual=[])
    with tqdm(range(sde.N)) as tqdm_setting:
        for i in range(sde.N):
            t = timesteps[i]
            
            '''method 1 (batched)'''
            vec_t = torch.ones(shape[0]*b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h -> (b n) t c h')       # (batch*b)*nf*(c+npara)*h*w

            '''corrector'''
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)     # (batch*b)*nf*(c+npara)*h*w

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h -> (b n) t c h')           
                      
            with torch.enable_grad():
                inp = temp.clone()                  # (batch*b)*nf*(c+npara)*h*w
                #print(inp.shape)
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)          # (batch*b)*nf*(c+npara)*h*w
                with torch.no_grad():
                    f, G = sde.discretize(temp, vec_t)
                    rev_f = f - G[:, None, None, None] ** 2 * score.detach() * (0.5 if probability_flow else 1.)
                    rev_G = torch.zeros_like(G) if probability_flow else G
                    temp_mean = temp - rev_f
                    temp_u = temp_mean + rev_G[:, None, None, None] * zb
                # dps loss
                _, std = sde.marginal_prob(xb, vec_t)
                if isinstance(sde, VESDE):
                    x0_hat = rearrange(std[:, None, None, None] ** 2 * score + inp, '(b n) t c h -> b n t c h', n=b)     # batch*b*nf*(c+npara)*h
                else:
                    alpha_sqrt_ = (1-std**2).sqrt()[:, None, None, None]
                    x0_hat = rearrange((std[:, None, None, None] ** 2 * score + inp)/alpha_sqrt_, '(b n) t c h -> b n t c h', n=b)     # batch*b*nf*(c+npara)*h
                x0_hat_temp = x_to_sample(x0_hat)
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())

                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_temp)) ** 2 / var).reshape(x0_hat.shape[0], -1), dim=-1)  # /scalar.sqrt()  *std[None, :, None, None]
                loss_dps = torch.sum(loss_dps, dim=0)  # /loss_dps.detach().mean().sqrt()
                if std_y is not None:
                    loss_dps = loss_dps/2.
                # loss_dps = loss_dps/loss_dps.detach().sqrt()    # normalize

                loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)    # , len(x0_hat[0])    /std[:, None, None].sqrt()
                loss_consis = torch.sum(loss_consis)        # *torch.softmax(loss_consis.detach(), dim=1)
                # loss_consis = loss_consis/loss_consis.detach().sqrt()    # normalize
                loss_consis_para = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, ncomp:]-x0_hat[:, 1:, :ol, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_consis_para = torch.sum(loss_consis_para, dim=0)
                # loss_consis_para = loss_consis_para/loss_consis_para.detach().sqrt()    # normalize
           
                #loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para

                if i >= (sde.N - residual_n_steps):
                    # 从x中提取声压数据（需根据实际数据结构调整维度）
                    # 假设x的形状为 [batch, b, nf, ncomp+config.num_conditions, image_size]
                    # 提取声压部分（前ncomp通道）并调整形状为 (T, N, C)
                    dxx = 0.02      # 传感器间距 [m]，根据实际情况填写
                    #dxx = 9/99      # 传感器间距 [m]，根据实际情况填写
                    #dtt = 1e-4      # 时间步长 [s]，例如采样率 100kHz → dt=1e-5
                    dtt = 1/12800      # 时间步长 [s]，例如采样率 100kHz → dt=1e-5
                    cc = 343.0      # 声速 [m/s] 
                    p = x_to_sample(rearrange(inp, '(b n) t c h -> b n t c h', n=b))[:, :, 0, :]
                    #print(p.shape)
                    p_tensor = p.to(device=device, dtype=dtype_torch)
                    residual = compute_acoustic_residual_torch(p_tensor, dxx, dtt, cc, source_term)
                    # 残差损失（理论残差应为0，最小化L2范数）
                    loss_residual = torch.norm(residual)
                    # 总损失 = 原损失 + 残差损失 * 权重
                    loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para + alpha_residual * loss_residual
                    #print(alpha_residual * loss_residual)
                    #print(loss_dps)
                    #print(loss_consis)
                    #print(loss_consis_para)
                else:
                    loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para


                tqdm_setting.set_description(f'loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e}')
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                temp = temp_u - dx     # (batch*b)*(nf*c+npara)*h*w
            #     # x = x_u
            temp = temp.detach()

            x = rearrange(temp, '(b n) t c h -> b n t c h', n=b)
            x_mean = rearrange(temp_mean, '(b n) t c h -> b n t c h', n=b)
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            tqdm_setting.update(1)
            #print(x_mean.shape)
            #print(x_to_sample(x_mean).shape)


    return x_to_sample(x_mean).detach().cpu().numpy(), x0_hats if save_sample_path else None, losses




def s3gm_sample_2d_ISS_ERA5(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=1000, reg_coef_pa = 1e4, reg_coef_ar = 1e4,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True):
    """
    Performs 2D inpainting and generation using a combination of parallel and
    autoregressive sampling, utilizing generate_parallel_2d_vesde_ddim.
    """
    if T_prime_y > 0:

        x_y, _, _ = generate_parallel_2d_ISS_ERA5(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                                    probability_flow=probability_flow, continuous=continuous,
                                                    denoising_steps=denoising_steps,eta=eta, reg_coef_pa = reg_coef_pa)
        '''
        x_y, _, _ = generate_parallel_2d(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                    alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                    snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                    probability_flow=probability_flow, continuous=continuous)
        '''
    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]

        x_extra, _ = generate_ar_2d_ISS_ERA5(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                    n_steps=n_steps, probability_flow=probability_flow,
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps, reg_coef_ar = reg_coef_ar)
        '''
        x_extra, _ = generate_ar_2d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                    n_steps=n_steps, probability_flow=probability_flow,
                                    alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                    continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        '''
        # Concatenate the generated extra part, excluding the overlapping portion from x_extra
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y

def generate_parallel_2d_ISS_ERA5(config, net, sde, y, transform, corrector, n_steps=5, 
                          alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                          gamma=1.e-2, num_steps=None,  
                          denoising_steps=100, 
                          overlap=1, eta=0.0, reg_coef_pa = 1e4,
                          device='cpu', dtype='float32', eps=1e-3, 
                          save_sample_path=False, probability_flow=False, 
                          continuous=True, data_scalar=None):
    
    dtype_torch = getattr(torch, dtype)
    
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
                                           corrector=corrector,
                                           continuous=continuous,
                                           snr=snr,
                                           n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))
    ns_real = b * (nf - ol) + ol 
    nc = config.num_conditions
    ds = denoising_steps
    reg_coef_pa = reg_coef_pa

    shape = [len(y), b, nf, ncomp + nc, config.image_size, config.image_size]
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size, config.image_size], 
                                dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    T_max = sde.T
    eps_min = eps
    
    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    #timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), x.size(5), device=x.device)
    
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[])
    
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    sde_start_step = ds - 5
    
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next
            
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')
            
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)

                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    x_pred = x_pred_mean + noise_coeff * noise

                    
                else:
                    with torch.no_grad():
                        f, G_val = sde.discretize(inp, vec_t)
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0

                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                
                x0_hat_reshaped = rearrange(x0_hat, '(b n) t c h w -> b n t c h w', n=b)
                x0_hat_sample = x_to_sample(x0_hat_reshaped)
                
                var = std_y**2 + gamma * sigma_current**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_sample))**2 / var).reshape(x0_hat_reshaped.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                loss_consis = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, :ncomp].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, :ncomp])**2
                )
                loss_consis = torch.sum(loss_consis, dim=-1).mean()
                
                loss_consis_para = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, ncomp:].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, ncomp:])**2
                )
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                #decay_factor = max(0.1, 1.0 - (i / ds)) 
                reg_loss = 1 * compute_regularization(x0_hat_sample, reg_coef_pa)
                
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para +  reg_loss
                #loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                losses['reg_loss'].append(reg_loss.item())
                
                mode = "ISS" if i < sde_start_step else "SDE"
                tqdm_setting.set_description(f'{mode} | loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e} | reg_loss.: { reg_loss.item():.5e}')
                
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8) 
                #momentum = 0.01 * momentum + 0.99 * dx
                #momentum = 0.99 * momentum + 0.01 * dx
                #x_pred = x_pred - momentum
                x_pred = x_pred - dx
            
            x_pred = x_pred.detach()
            #print(x_pred.shape)
            x = rearrange(x_pred, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(x_pred if i < sde_start_step else temp_mean,  # SDE没有mean，直接使用x_pred
                              '(b n) t c h w -> b n t c h w', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses


def generate_ar_2d_ISS_ERA5(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1., mult=10, transform_init=None, num_steps=10, overlap=1,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3, reg_coef_ar = 1e4):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            )
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    ds = sde.N
    reg_coef_ar = reg_coef_ar
    T_max = sde.T
    eps_min = eps

    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    #timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    sde_start_step = ds - 10
    #sde_start_step = 0
    
    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(x0), nf, ncomp+config.num_conditions, config.image_size, config.image_size]       # batch*nf*(c+npara)*h*w

    transform_ar, alpha_ar = lambda x: x[:, :ol], alpha
    transform_init, alpha_init = transform_ar if transform_init is None else transform_init, alpha*mult

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        return reg_loss
    
    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
            transform = transform_init
            alpha = alpha_init
        else:
            y = x_mean[:, -ol:].detach()
            transform = transform_ar
            alpha = alpha_ar
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(ds)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next  
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)

                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]

                x0_hat = inp + (sigma_current ** 2) * score
            
                if i < sde_start_step:
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    x_u = x_pred_mean
                    
                else:
                    with torch.no_grad():
                        '''
                        f, G_val = sde.discretize(inp, vec_t)
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0
                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                        x_u = x_pred
                        '''
                        f, G = sde.discretize(x, vec_t)
                        rev_f = f - G[:, None, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                        rev_G = torch.zeros_like(G) if probability_flow else G
                        x_mean = x - rev_f
                        x_u = x_mean + rev_G[:, None, None, None, None] * z
                
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                #x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                reg_loss = 1 * compute_regularization(x0_hat, reg_coef_ar)
                loss = alpha * loss_dps + reg_loss
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated











































def s3gm_sample_2d_ISS_2dsound(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=1000, reg_coef_pa = 1e4, reg_coef_ar = 1e4,
                                use_residual_objective=False,
                                data_scalar=None,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True):

    """
    Performs 2D inpainting and generation using a combination of parallel and
    autoregressive sampling, utilizing generate_parallel_2d_vesde_ddim.
    """
    if T_prime_y > 0:

        x_y, _, _ = generate_parallel_2d_ISS_2dsound(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                        alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                        snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                        probability_flow=probability_flow, continuous=continuous,
                                        denoising_steps=denoising_steps,eta=eta, reg_coef_pa = reg_coef_pa)
        '''
        x_y, _, _ = generate_parallel_2d(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                        alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                        snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                        probability_flow=probability_flow, continuous=continuous)
        '''
    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]


        x_extra, _ = generate_ar_2d_ISS_2dsound(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps, reg_coef_ar = reg_coef_ar)
        '''
        x_extra, _ = generate_ar_2d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        '''
        # Concatenate the generated extra part, excluding the overlapping portion from x_extra
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y

def generate_parallel_2d_ISS_2dsound(config, net, sde, y, transform, corrector, n_steps=5, 
                          alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                          gamma=1.e-2, num_steps=None,  
                          denoising_steps=100, 
                          overlap=1, eta=0.0, reg_coef_pa = 1e4,
                          device='cpu', dtype='float32', eps=1e-3, 
                          save_sample_path=False, probability_flow=False, 
                          continuous=True, data_scalar=None):
    
    dtype_torch = getattr(torch, dtype)
    
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
                                           corrector=corrector,
                                           continuous=continuous,
                                           snr=snr,
                                           n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))
    ns_real = b * (nf - ol) + ol 
    nc = config.num_conditions
    ds = denoising_steps
    reg_coef_pa = reg_coef_pa

    shape = [len(y), b, nf, ncomp + nc, config.image_size, config.image_size]
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size, config.image_size], 
                                dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    T_max = sde.T
    eps_min = eps
    
    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    #timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        #reg_loss = lambda_reg * dx_t.mean() + 1e2 * (dx_h.mean() + dx_w.mean()) 
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), x.size(5), device=x.device)
    
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[])
    
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    sde_start_step = ds - 5
    
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next
            
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')
            
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)

                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    x_pred = x_pred_mean + noise_coeff * noise

                    
                else:
                    with torch.no_grad():
                        f, G_val = sde.discretize(inp, vec_t)
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0

                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                
                x0_hat_reshaped = rearrange(x0_hat, '(b n) t c h w -> b n t c h w', n=b)
                x0_hat_sample = x_to_sample(x0_hat_reshaped)
                
                var = std_y**2 + gamma * sigma_current**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_sample))**2 / var).reshape(x0_hat_reshaped.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                loss_consis = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, :ncomp].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, :ncomp])**2
                )
                loss_consis = torch.sum(loss_consis, dim=-1).mean()
                
                loss_consis_para = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, ncomp:].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, ncomp:])**2
                )
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                #decay_factor = max(0.1, 1.0 - (i / ds)) 
                reg_loss = 1 * compute_regularization(x0_hat_sample, reg_coef_pa)
                
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para +  reg_loss
                #loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                losses['reg_loss'].append(reg_loss.item())
                
                mode = "ISS" if i < sde_start_step else "SDE"
                tqdm_setting.set_description(f'{mode} | loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e} | reg_loss.: { reg_loss.item():.5e}')
                
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8) 
                #momentum = 0.01 * momentum + 0.99 * dx
                #momentum = 0.99 * momentum + 0.01 * dx
                #x_pred = x_pred - momentum
                x_pred = x_pred - dx
            
            x_pred = x_pred.detach()
            #print(x_pred.shape)
            x = rearrange(x_pred, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(x_pred if i < sde_start_step else temp_mean,  # SDE没有mean，直接使用x_pred
                              '(b n) t c h w -> b n t c h w', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    '''
    # ========================= 【只保存 loss_dps】 =========================
    try:
        import pandas as pd
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        csv_name = f"loss_dps_ISS_{timestamp}.csv"
        
        # 直接指定只存 loss_dps，简单粗暴，绝不报错
        data_to_save = {
            'step': range(1, len(losses['loss_dps']) + 1),
            'loss_dps': losses['loss_dps']
        }
        
        df_loss = pd.DataFrame(data_to_save)
        df_loss.to_csv(csv_name, index=False)
        print(f"\n>>> [表格已保存] ISS loss_dps 数据已存入: {csv_name}")
        
    except Exception as e:
        print(f"\n>>> [保存失败]: {e}")
    # =====================================================================   
    '''
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses


def generate_ar_2d_ISS_2dsound(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1., mult=10, transform_init=None, num_steps=10, overlap=1,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3, reg_coef_ar = 1e4):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            )
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    ds = sde.N
    reg_coef_ar = reg_coef_ar
    T_max = sde.T
    eps_min = eps

    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    #timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    sde_start_step = ds - 10
    #sde_start_step = 0
    
    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(x0), nf, ncomp+config.num_conditions, config.image_size, config.image_size]       # batch*nf*(c+npara)*h*w

    transform_ar, alpha_ar = lambda x: x[:, :ol], alpha
    transform_init, alpha_init = transform_ar if transform_init is None else transform_init, alpha*mult

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * dx_t.mean() + 1e3*(dx_h.mean() + dx_w.mean()) 
        return reg_loss
    
    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
            transform = transform_init
            alpha = alpha_init
        else:
            y = x_mean[:, -ol:].detach()
            transform = transform_ar
            alpha = alpha_ar
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(ds)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next  
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)

                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]

                x0_hat = inp + (sigma_current ** 2) * score
            
                if i < sde_start_step:
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    x_u = x_pred_mean
                    
                else:
                    with torch.no_grad():
                        '''
                        f, G_val = sde.discretize(inp, vec_t)
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0
                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                        x_u = x_pred
                        '''
                        f, G = sde.discretize(x, vec_t)
                        rev_f = f - G[:, None, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                        rev_G = torch.zeros_like(G) if probability_flow else G
                        x_mean = x - rev_f
                        x_u = x_mean + rev_G[:, None, None, None, None] * z
                
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                #x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                reg_loss = 1 * compute_regularization(x0_hat, reg_coef_ar)
                loss = alpha * loss_dps + reg_loss
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated














import torch
import torch.nn.functional as F

def calculate_physics_residual_loss(x_unnorm, c=343.0, rho0=1.21, dt=1e-4, dx=0.01, dy=0.01):
    """
    计算2D声学场的物理残差损失（波动方程、连续性、欧拉方程）。
    
    参数:
    x_unnorm (torch.Tensor): 逆标准化后的样本，shape (b, t, c, h, w)
                             c=0: 声压 (p)
                             c=1: x方向振速 (vx)
                             c=2: y方向振速 (vy)
    c (float): 声速
    rho0 (float): 介质密度
    dt (float): 时间步长
    dx (float): x方向空间步长
    dy (float): y方向空间步长

    返回:
    loss_wave, loss_cont, loss_euler (torch.Tensor)
    """
    
    # 确保有足够的时间步和空间点来进行二阶中心差分
    if x_unnorm.shape[1] < 3 or x_unnorm.shape[3] < 3 or x_unnorm.shape[4] < 3:
        return torch.tensor(0.0, device=x_unnorm.device), \
               torch.tensor(0.0, device=x_unnorm.device), \
               torch.tensor(0.0, device=x_unnorm.device)

    # 分离通道
    p  = x_unnorm[:, :, 0:1, :, :]  # (b, t, 1, h, w)
    vx = x_unnorm[:, :, 1:2, :, :]
    vy = x_unnorm[:, :, 2:3, :, :]

    # --- 1. 波动方程 (来自图像1) ---
    # R_wave = d2p/dt2 - c^2 * (d2p/dx2 + d2p/dy2)
    
    # 中心切片 (t, x, y)
    p_center = p[:, 1:-1, :, 1:-1, 1:-1]
    
    # d2p/dt2
    p_tp1 = p[:, 2:, :, 1:-1, 1:-1]
    p_tm1 = p[:, :-2, :, 1:-1, 1:-1]
    d2p_dt2 = (p_tp1 - 2*p_center + p_tm1) / (dt**2)
    
    # d2p/dx2
    p_xp1 = p[:, 1:-1, :, 2:, 1:-1]
    p_xm1 = p[:, 1:-1, :, :-2, 1:-1]
    d2p_dx2 = (p_xp1 - 2*p_center + p_xm1) / (dx**2)
    
    # d2p/dy2
    p_yp1 = p[:, 1:-1, :, 1:-1, 2:]
    p_ym1 = p[:, 1:-1, :, 1:-1, :-2]
    d2p_dy2 = (p_yp1 - 2*p_center + p_ym1) / (dy**2)
    
    laplacian_p = d2p_dx2 + d2p_dy2
    R_wave = d2p_dt2 - (c**2) * laplacian_p
    loss_wave = torch.mean(R_wave**2)

    # --- 2. 连续性方程 (来自图像2) ---
    # R_cont = (1/(rho0*c^2)) * dp/dt + (dvx/dx + dvy/dy)
    
    # dp/dt (中心差分)
    dp_dt = (p[:, 2:, :, 1:-1, 1:-1] - p[:, :-2, :, 1:-1, 1:-1]) / (2 * dt)
    
    # dvx/dx (中心差分)
    vx_xp1 = vx[:, 1:-1, :, 2:, 1:-1]
    vx_xm1 = vx[:, 1:-1, :, :-2, 1:-1]
    dvx_dx = (vx_xp1 - vx_xm1) / (2 * dx)
    
    # dvy/dy (中心差分)
    vy_yp1 = vy[:, 1:-1, :, 1:-1, 2:]
    vy_ym1 = vy[:, 1:-1, :, 1:-1, :-2]
    dvy_dy = (vy_yp1 - vy_ym1) / (2 * dy)
    
    R_cont = (1 / (rho0 * c**2)) * dp_dt + dvx_dx + dvy_dy
    loss_cont = torch.mean(R_cont**2)

    # --- 3. 欧拉方程 (来自图像2) ---
    # R_euler_x = rho0 * dvx/dt + dp/dx
    # R_euler_y = rho0 * dvy/dt + dp/dy
    
    # dvx/dt
    dvx_dt = (vx[:, 2:, :, 1:-1, 1:-1] - vx[:, :-2, :, 1:-1, 1:-1]) / (2 * dt)
    
    # dvy/dt
    dvy_dt = (vy[:, 2:, :, 1:-1, 1:-1] - vy[:, :-2, :, 1:-1, 1:-1]) / (2 * dt)
    
    # dp/dx
    dp_dx = (p[:, 1:-1, :, 2:, 1:-1] - p[:, 1:-1, :, :-2, 1:-1]) / (2 * dx)
    
    # dp/dy
    dp_dy = (p[:, 1:-1, :, 1:-1, 2:] - p[:, 1:-1, :, 1:-1, :-2]) / (2 * dy)
    
    R_euler_x = rho0 * dvx_dt + dp_dx
    R_euler_y = rho0 * dvy_dt + dp_dy
    
    loss_euler = torch.mean(R_euler_x**2) + torch.mean(R_euler_y**2)

    return loss_wave, loss_cont, loss_euler





import os
import functools
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from einops import rearrange


def s3gm_sample_2d_ISS_2dsound_res(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=1000, reg_coef_pa = 1e4, reg_coef_ar = 1e4,
                                use_residual_objective=False,
                                data_scalar=None,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True,
                                physics_start_step=20,
                                reg_coef_wave=0.1, 
                                reg_coef_cont=1.0, 
                                reg_coef_euler=100, 
                                physics_c=343.0, 
                                physics_rho0=1.21, 
                                physics_dt=2e-4,
                                physics_dx=1/21,
                                physics_dy=1/21, 
                               ):

    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_2d_ISS_2dsound_res(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                        alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                        snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                        probability_flow=probability_flow, continuous=continuous,
                                        denoising_steps=denoising_steps,eta=eta, reg_coef_pa = reg_coef_pa, 
                                        data_scalar=data_scalar,
                                        physics_start_step=physics_start_step,
                                        reg_coef_wave=reg_coef_wave,
                                        reg_coef_cont=reg_coef_cont,
                                        reg_coef_euler=reg_coef_euler,
                                        physics_c=physics_c,
                                        physics_rho0=physics_rho0,
                                        physics_dt=physics_dt,
                                        physics_dx=physics_dx,
                                        physics_dy=physics_dy
                                        )

    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]

        x_extra, _ = generate_ar_2d_ISS_2dsound_res(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps, reg_coef_ar = reg_coef_ar,
                                     data_scalar=data_scalar,
                                     physics_start_step=physics_start_step,
                                     reg_coef_wave=reg_coef_wave,
                                     reg_coef_cont=reg_coef_cont,
                                     reg_coef_euler=reg_coef_euler,
                                     physics_c=physics_c,
                                     physics_rho0=physics_rho0,
                                     physics_dt=physics_dt,
                                     physics_dx=physics_dx,
                                     physics_dy=physics_dy,
                                                   )
        '''
        x_extra, _ = generate_ar_2d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        '''
        # Concatenate the generated extra part, excluding the overlapping portion from x_extra
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y



def generate_parallel_2d_ISS_2dsound_res(config, net, sde, y, transform, corrector, n_steps=5, 
                                        alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                                        gamma=1.e-2, num_steps=None,  
                                        denoising_steps=100, 
                                        overlap=1, eta=0.0, reg_coef_pa = 1e4,
                                        device='cpu', dtype='float32', eps=1e-3, 
                                        save_sample_path=False, probability_flow=False, 
                                        continuous=True, 
                                        data_scalar=None,
                                        physics_start_step=20,
                                        reg_coef_wave=0.1,
                                        reg_coef_cont=1.0,
                                        reg_coef_euler=100,
                                        physics_c=343.0,
                                        physics_rho0=1.21, 
                                        physics_dt=2e-4,
                                        physics_dx=1/21,
                                        physics_dy=1/21
                                        ):

                                        
    dtype_torch = getattr(torch, dtype)

    
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))
    ns_real = b * (nf - ol) + ol 
    nc = config.num_conditions
    ds = denoising_steps
    reg_coef_pa = reg_coef_pa

    shape = [len(y), b, nf, ncomp + nc, config.image_size, config.image_size]
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size, config.image_size], 
                                    dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    T_max = sde.T
    eps_min = eps
    
    # ... (adaptive_cosine_timesteps 定义，如果您使用它)
    
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), x.size(5), device=x.device)
    

    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[],
                  loss_phy=[], loss_wave=[], loss_cont=[], loss_euler=[])
    
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    sde_start_step = ds - 5
    
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next
            
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')
            
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)

                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    x_pred = x_pred_mean + noise_coeff * noise

                    
                else:
                    with torch.no_grad():
                        f, G_val = sde.discretize(inp, vec_t)
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0

                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                
                x0_hat_reshaped = rearrange(x0_hat, '(b n) t c h w -> b n t c h w', n=b)
                x0_hat_sample = x_to_sample(x0_hat_reshaped)
                
                var = std_y**2 + gamma * sigma_current**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_sample))**2 / var).reshape(x0_hat_reshaped.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                loss_consis = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, :ncomp].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, :ncomp])**2
                )
                loss_consis = torch.sum(loss_consis, dim=-1).mean()
                
                loss_consis_para = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, ncomp:].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, ncomp:])**2
                )
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                reg_loss = 1 * compute_regularization(x0_hat_sample, reg_coef_pa)
                
                # <--- MODIFICATION: 计算物理损失 ---
                loss_phy = torch.tensor(0.0, device=x0_hat_sample.device)
                loss_wave = torch.tensor(0.0)
                loss_cont = torch.tensor(0.0)
                loss_euler = torch.tensor(0.0)
                
                # 只在最后 N 步计算物理损失
                if (ds - i) <= physics_start_step:
                    if data_scalar is not None:
                        # 检查通道数是否足够 (p, vx, vy)
                        if ncomp >= 3:
                            # 逆标准化 (只对分量)
                            x0_hat_unnorm = data_scalar(x0_hat_sample[:, :, :ncomp])
                            
                            loss_wave, loss_cont, loss_euler = calculate_physics_residual_loss(
                                x0_hat_unnorm, 
                                c=physics_c, 
                                rho0=physics_rho0, 
                                dt=physics_dt, 
                                dx=physics_dx, 
                                dy=physics_dy
                            )
                            
                            loss_phy = (reg_coef_wave * loss_wave + 
                                        reg_coef_cont * loss_cont + 
                                        reg_coef_euler * loss_euler)
                        else:
                            # 如果通道不够，在第一步打印警告
                            if (ds - i) == physics_start_step:
                                print(f"Warning: Physics loss skipped. Expected ncomp >= 3, but got {ncomp}.")
                    else:
                        # 如果没有提供 data_scalar，在第一步打印警告
                        if (ds - i) == physics_start_step:
                            print("Warning: Physics loss skipped. 'data_scalar' (scalar_inv) not provided.")
                # <--- MODIFICATION END ---
                
                
                # <--- MODIFICATION: 将 loss_phy 添加到总损失
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para + reg_loss + 1e-3 * loss_phy
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                losses['reg_loss'].append(reg_loss.item())
                # <--- MODIFICATION: 记录新的损失 ---
                losses['loss_phy'].append(loss_phy.item())
                losses['loss_wave'].append(loss_wave.item())
                losses['loss_cont'].append(loss_cont.item())
                losses['loss_euler'].append(loss_euler.item())
                
                mode = "ISS" if i < sde_start_step else "SDE"
                
                # <--- MODIFICATION: 更新 tqdm 描述 ---
                tqdm_desc = (
                    f'{mode} | loss total: {loss.item():.5e} | '
                    f'loss obs.: {alpha * loss_dps.item():.5e} | '
                    f'loss consis.: {beta1 * loss_consis.item():.5e} | '
                    f'reg_loss.: {reg_loss.item():.5e} | '
                    f'loss_phy: {loss_phy.item():.5e}'
                )
                tqdm_setting.set_description(tqdm_desc)
                
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8) 
                
                x_pred = x_pred - dx
            
            x_pred = x_pred.detach()
            x = rearrange(x_pred, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(x_pred if i < sde_start_step else temp_mean,
                               '(b n) t c h w -> b n t c h w', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses



def generate_ar_2d_ISS_2dsound_res(config, net, sde, predictor, corrector, shape, snr, x0=None, n_steps=1, probability_flow=False, alpha=1., mult=10, transform_init=None, num_steps=10, overlap=1,
               continuous=False, device='cpu', denoise=True, dtype='float32', eps=1e-3, reg_coef_ar = 1e4,
               data_scalar=None,
               physics_start_step=20,
               reg_coef_wave=0.1,
               reg_coef_cont=1.0,
               reg_coef_euler=100,
               physics_c=343.0,
               physics_rho0=1.21,
               physics_dt=2e-4,
               physics_dx=1/21,
               physics_dy=1/21):
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    net_fn = lambda a, b: predict_fn(net, sde, a, b)
    x0 = torch.tensor(x0, device=device).float()            # batch*ol*(c+npara)*h*w
    ds = sde.N
    reg_coef_ar = reg_coef_ar
    T_max = sde.T
    eps_min = eps

    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    #timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    sde_start_step = ds - 10
    #sde_start_step = 0
    
    nf = config.num_frames
    ns = num_steps
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape = [len(x0), nf, ncomp+config.num_conditions, config.image_size, config.image_size]       # batch*nf*(c+npara)*h*w

    transform_ar, alpha_ar = lambda x: x[:, :ol], alpha
    transform_init, alpha_init = transform_ar if transform_init is None else transform_init, alpha*mult

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * dx_t.mean() + 1e3*(dx_h.mean() + dx_w.mean()) 
        return reg_loss
    
    x_generated = []
    pred = []
    for i_b in range(b):
        if i_b == 0:
            y = x0
            transform = transform_init
            alpha = alpha_init
        else:
            y = x_mean[:, -ol:].detach()
            transform = transform_ar
            alpha = alpha_ar
        x = sde.prior_sampling(shape).to(device).float()
        for i in tqdm(range(ds)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next  
            vec_t = torch.ones(shape[0], device=t.device).float() * t
            '''corrector'''
            x, x_mean = corrector_update_fn(x, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            with torch.enable_grad():
                inp = x.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)

                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]

                x0_hat = inp + (sigma_current ** 2) * score
            
                if i < sde_start_step:
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    x_u = x_pred_mean
                    
                else:
                    with torch.no_grad():
                        '''
                        f, G_val = sde.discretize(inp, vec_t)
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0
                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                        x_u = x_pred
                        '''
                        f, G = sde.discretize(x, vec_t)
                        rev_f = f - G[:, None, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
                        rev_G = torch.zeros_like(G) if probability_flow else G
                        x_mean = x - rev_f
                        x_u = x_mean + rev_G[:, None, None, None, None] * z
                
                    # x_u = x
                # dps loss
                _, std = sde.marginal_prob(x, t)
                #x0_hat = std**2*score + inp
                loss_dps = ((y-transform(x0_hat))**2).sum()
                reg_loss = 1 * compute_regularization(x0_hat, reg_coef_ar)
                
                x0_hat_sample = x0_hat
                loss_phy = torch.tensor(0.0, device=x0_hat_sample.device)
                loss_wave = torch.tensor(0.0)
                loss_cont = torch.tensor(0.0)
                loss_euler = torch.tensor(0.0)
                
                # 只在最后 N 步计算物理损失
                if (ds - i) <= physics_start_step:
                    if data_scalar is not None:
                        # 检查通道数是否足够 (p, vx, vy)
                        if ncomp >= 3:
                            # 逆标准化 (只对分量)
                            x0_hat_unnorm = data_scalar(x0_hat_sample[:, :, :ncomp])
                            
                            loss_wave, loss_cont, loss_euler = calculate_physics_residual_loss(
                                x0_hat_unnorm, 
                                c=physics_c, 
                                rho0=physics_rho0, 
                                dt=physics_dt, 
                                dx=physics_dx, 
                                dy=physics_dy
                            )
                            
                            loss_phy = (reg_coef_wave * loss_wave + 
                                        reg_coef_cont * loss_cont + 
                                        reg_coef_euler * loss_euler)
                        else:
                            # 如果通道不够，在第一步打印警告
                            if (ds - i) == physics_start_step:
                                print(f"Warning: Physics loss skipped. Expected ncomp >= 3, but got {ncomp}.")
                    else:
                        # 如果没有提供 data_scalar，在第一步打印警告
                        if (ds - i) == physics_start_step:
                            print("Warning: Physics loss skipped. 'data_scalar' (scalar_inv) not provided.")
                # <--- MODIFICATION END ---


                
                loss = alpha * loss_dps + reg_loss + 1e-3 * loss_phy
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                x = x_u - dx       # /torch.sqrt(scalar1.mean())
                # x = x_u
            x = x.detach()
        pred.append(x_mean if i_b==0 else x_mean[:, ol:])
    return torch.cat(pred, dim=1).detach().cpu().numpy() if denoise else x, x_generated










# ==========================================
# 1. 辅助函数：计算一维声学物理残差
# ==========================================
def calculate_physics_residual_loss_1d(data, c=343.0, rho0=1.21, dt=2e-4, dx=1/21):
    """
    计算一维声学方程残差
    data: [Batch, Time, Channels, Space] (未归一化的物理量)
    假设: Channel 0 = Pressure (p), Channel 1 = Velocity (v)
    """
    p = data[:, :, 0, :] # [B, T, H]
    v = data[:, :, 1, :] # [B, T, H]

    # --- 1. 计算偏导数 (使用有限差分) ---
    # 时间导数 (Time derivatives) [:, 1:, :] - [:, :-1, :] -> 形状减少 1 帧
    dp_dt = (p[:, 1:, :] - p[:, :-1, :]) / dt
    dv_dt = (v[:, 1:, :] - v[:, :-1, :]) / dt

    # 空间导数 (Space derivatives) [:, :, 1:] - [:, :, :-1] -> 形状减少 1 个空间点
    dp_dx = (p[:, :, 1:] - p[:, :, :-1]) / dx
    dv_dx = (v[:, :, 1:] - v[:, :, :-1]) / dx

    # --- 2. 对齐维度 ---
    # 为了让 dt 和 dx 的张量形状匹配，我们需要取交集部分
    # dp_dt 缺少最后一帧，dp_dx 缺少最后一个空间点
    # 我们统一截取到 [:, 0:-1, 0:-1] 的范围 (或者根据具体差分格式调整，这里采用最简对齐)
    
    # 对齐时间项 (去掉最后一个空间点)
    dp_dt_c = dp_dt[:, :, :-1]
    dv_dt_c = dv_dt[:, :, :-1]
    
    # 对齐空间项 (去掉最后一帧)
    dp_dx_c = dp_dx[:, :-1, :]
    dv_dx_c = dv_dx[:, :-1, :]

    # --- 3. 计算物理残差 ---
    # 连续性方程: dp/dt + rho0 * c^2 * dv/dx = 0
    res_cont = dp_dt_c + rho0 * (c**2) * dv_dx_c
    
    # 动量方程 (Euler): rho0 * dv/dt + dp/dx = 0
    res_euler = rho0 * dv_dt_c + dp_dx_c

    # --- 4. 计算 MSE Loss ---
    loss_cont = torch.mean(res_cont ** 2)
    loss_euler = torch.mean(res_euler ** 2)
    
    # 简单的波方程项 (可选，作为辅助)
    loss_wave = loss_cont + loss_euler 

    return loss_wave, loss_cont, loss_euler






def calculate_physics_residual_loss_1d_2(data, c=343.0, rho0=1.21, dt=2e-4, dx=1/21):
    """
    计算一维声学方程残差 (使用二阶中心差分，包含波动方程)
    data: [Batch, Time, Channels, Space] (未归一化的物理量)
    Channel 0 = Pressure (p)
    Channel 1 = Velocity (v)
    """
    # 确保有足够的时间步和空间点来进行二阶中心差分 (至少需要3个点)
    if data.shape[1] < 3 or data.shape[3] < 3:
        return torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device)

    p = data[:, :, 0:1, :] # [B, T, 1, H]
    v = data[:, :, 1:2, :] # [B, T, 1, H]

    # ==========================================
    # 1. 波动方程 (Wave Equation) - 二阶
    # d2p/dt2 - c^2 * d2p/dx2 = 0
    # ==========================================
    
    # 中心点 (t, x) 范围: [1:-1, 1:-1]
    p_center = p[:, 1:-1, :, 1:-1]

    # 时间二阶导 d2p/dt2
    p_tp1 = p[:, 2:, :, 1:-1]   # t+1
    p_tm1 = p[:, :-2, :, 1:-1]  # t-1
    d2p_dt2 = (p_tp1 - 2*p_center + p_tm1) / (dt**2)

    # 空间二阶导 d2p/dx2
    p_xp1 = p[:, 1:-1, :, 2:]   # x+1
    p_xm1 = p[:, 1:-1, :, :-2]  # x-1
    d2p_dx2 = (p_xp1 - 2*p_center + p_xm1) / (dx**2)

    # 残差
    res_wave = d2p_dt2 - (c**2) * d2p_dx2
    loss_wave = torch.mean(res_wave**2)

    # ==========================================
    # 2. 连续性方程 (Continuity) - 一阶
    # (1 / (rho0 * c^2)) * dp/dt + dv/dx = 0
    # ==========================================

    # 时间一阶导 dp/dt (中心差分)
    dp_dt = (p[:, 2:, :, 1:-1] - p[:, :-2, :, 1:-1]) / (2 * dt)

    # 空间一阶导 dv/dx (中心差分)
    # 注意：这里取 v 在 (t) 时刻的导数，对应 p 的中心时刻
    v_xp1 = v[:, 1:-1, :, 2:]
    v_xm1 = v[:, 1:-1, :, :-2]
    dv_dx = (v_xp1 - v_xm1) / (2 * dx)

    res_cont = (1 / (rho0 * c**2)) * dp_dt + dv_dx
    loss_cont = torch.mean(res_cont**2)

    # ==========================================
    # 3. 动量方程 (Euler / Momentum) - 一阶
    # rho0 * dv/dt + dp/dx = 0
    # ==========================================

    # 时间一阶导 dv/dt (中心差分)
    dv_dt = (v[:, 2:, :, 1:-1] - v[:, :-2, :, 1:-1]) / (2 * dt)

    # 空间一阶导 dp/dx (中心差分)
    p_xp1_euler = p[:, 1:-1, :, 2:]
    p_xm1_euler = p[:, 1:-1, :, :-2]
    dp_dx = (p_xp1_euler - p_xm1_euler) / (2 * dx)

    res_euler = rho0 * dv_dt + dp_dx
    loss_euler = torch.mean(res_euler**2)

    return loss_wave, loss_cont, loss_euler



    

def calculate_physics_residual_loss_1d_3(data, c=343.0, rho0=1.21, dt=1e-5, dx=0.032, use_huber=True):
    """
    计算一维声学方程残差 (归一化版 + Huber Loss) + 硬声场边界损失
    """
    # 确保有足够的时间步和空间点
    if data.shape[1] < 3 or data.shape[3] < 3:
        return torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device) # 增加一个返回值占位

    p = data[:, :, 0:1, :] # [B, T, 1, H]
    v = data[:, :, 1:2, :] # [B, T, 1, H]

    # 定义损失函数：Huber Loss 对异常值更鲁棒，防止梯度爆炸
    criterion = torch.nn.SmoothL1Loss() if use_huber else torch.nn.MSELoss()

    # ==========================================
    # 1. 波动方程 (Wave Equation)
    # ==========================================
    p_center = p[:, 1:-1, :, 1:-1]
    
    # 时间二阶导
    d2p_dt2 = (p[:, 2:, :, 1:-1] - 2*p_center + p[:, :-2, :, 1:-1]) / (dt**2)
    # 空间二阶导
    d2p_dx2 = (p[:, 1:-1, :, 2:] - 2*p_center + p[:, 1:-1, :, :-2]) / (dx**2)

    # 计算归一化残差
    res_wave = (1.0 / (c**2)) * d2p_dt2 - d2p_dx2
    loss_wave = criterion(res_wave, torch.zeros_like(res_wave))

    # ==========================================
    # 2. 连续性方程 (Continuity)
    # ==========================================
    dp_dt = (p[:, 2:, :, 1:-1] - p[:, :-2, :, 1:-1]) / (2 * dt)
    
    v_xp1 = v[:, 1:-1, :, 2:]
    v_xm1 = v[:, 1:-1, :, :-2]
    dv_dx = (v_xp1 - v_xm1) / (2 * dx)

    res_cont = (1.0 / (rho0 * (c**2))) * dp_dt + dv_dx
    loss_cont = criterion(res_cont, torch.zeros_like(res_cont))

    # ==========================================
    # 3. 动量方程 (Euler)
    # ==========================================
    dv_dt = (v[:, 2:, :, 1:-1] - v[:, :-2, :, 1:-1]) / (2 * dt)
    
    p_xp1 = p[:, 1:-1, :, 2:]
    p_xm1 = p[:, 1:-1, :, :-2]
    dp_dx = (p_xp1 - p_xm1) / (2 * dx)

    res_euler = dv_dt + (1.0 / rho0) * dp_dx
    loss_euler = criterion(res_euler, torch.zeros_like(res_euler))

    # ==========================================
    # 4. [新增] 硬声场边界 (Rigid Wall BC)
    # 位置: 最右端 (index -1)
    # 条件: 振速 v = 0
    # ==========================================
    v_outlet = v[:, :, :, -1] # 取出最右端的振速数据 [B, T, 1]
    #v_outlet = v[:, :, :, 0] # 取出最右端的振速数据 [B, T, 1]
    
    loss_bc = criterion(v_outlet, torch.zeros_like(v_outlet))

    return loss_wave, loss_cont, loss_euler, loss_bc





def calculate_physics_residual_loss_1d_4(data, c=343.0, rho0=1.21, dt=2e-4, dx=1/21, smooth=True):
    """
    计算一维声学方程残差 (带平滑预处理 - 修正版)
    data: [Batch, Time, Channels, Space]
    """
    # 确保有足够的时间步和空间点
    if data.shape[1] < 3 or data.shape[3] < 3:
        return torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device)

    # -------------------------------------------------------
    # 0. 平滑预处理 (关键步骤!)
    # -------------------------------------------------------
    if smooth:
        # 定义一个简单的 1D 高斯核 [1, 2, 1] / 4
        # F.conv1d 的权重形状要求: (Out_channels, In_channels, Kernel_size)
        # 这里我们需要 (1, 1, 3)
        kernel = torch.tensor([[[1., 2., 1.]]], device=data.device) / 4.0
        
        # 【修正】：不要在这里执行 unsqueeze(1)，上面定义的时候已经是 3维 [1,1,3] 了
        # kernel = kernel.unsqueeze(1)  <-- 删掉这行
        
        # 对 Time 维度进行 reshape 以便对 Space 维度做卷积
        # data: [B, T, C, H]
        b, t, ch, h = data.shape
        data_reshaped = data.view(b * t, ch, h) # [Batch*Time, Channels, Space]
        
        # 分别平滑 p (channel 0) 和 v (channel 1)
        # data_reshaped[:, 0:1, :] shape is [N, 1, L] -> 满足 conv1d 输入要求
        p_smooth = F.conv1d(data_reshaped[:, 0:1, :], kernel, padding=1)
        v_smooth = F.conv1d(data_reshaped[:, 1:2, :], kernel, padding=1)
        
        # 还原形状 [B, T, 1, H]
        p = p_smooth.view(b, t, 1, h)
        v = v_smooth.view(b, t, 1, h)
    else:
        p = data[:, :, 0:1, :] 
        v = data[:, :, 1:2, :] 

    # 定义损失函数 (使用 Huber Loss 防止梯度爆炸)
    criterion = torch.nn.SmoothL1Loss()

    # -------------------------------------------------------
    # 1. 波动方程 (归一化)
    # (1/c^2) * d2p/dt2 - d2p/dx2 = 0
    # -------------------------------------------------------
    p_center = p[:, 1:-1, :, 1:-1]
    
    # 时间二阶导
    d2p_dt2 = (p[:, 2:, :, 1:-1] - 2*p_center + p[:, :-2, :, 1:-1]) / (dt**2)
    # 空间二阶导
    d2p_dx2 = (p[:, 1:-1, :, 2:] - 2*p_center + p[:, 1:-1, :, :-2]) / (dx**2)

    res_wave = (1.0 / (c**2)) * d2p_dt2 - d2p_dx2
    loss_wave = criterion(res_wave, torch.zeros_like(res_wave))

    # -------------------------------------------------------
    # 2. 连续性方程 (归一化)
    # (1 / (rho0 * c^2)) * dp/dt + dv/dx = 0
    # -------------------------------------------------------
    dp_dt = (p[:, 2:, :, 1:-1] - p[:, :-2, :, 1:-1]) / (2 * dt)
    
    v_xp1 = v[:, 1:-1, :, 2:]
    v_xm1 = v[:, 1:-1, :, :-2]
    dv_dx = (v_xp1 - v_xm1) / (2 * dx)

    res_cont = (1.0 / (rho0 * (c**2))) * dp_dt + dv_dx
    loss_cont = criterion(res_cont, torch.zeros_like(res_cont))

    # -------------------------------------------------------
    # 3. 动量方程 (归一化)
    # dv/dt + (1/rho0) * dp/dx = 0
    # -------------------------------------------------------
    dv_dt = (v[:, 2:, :, 1:-1] - v[:, :-2, :, 1:-1]) / (2 * dt)
    
    p_xp1 = p[:, 1:-1, :, 2:]
    p_xm1 = p[:, 1:-1, :, :-2]
    dp_dx = (p_xp1 - p_xm1) / (2 * dx)

    res_euler = dv_dt + (1.0 / rho0) * dp_dx
    loss_euler = criterion(res_euler, torch.zeros_like(res_euler))

    return loss_wave, loss_cont, loss_euler



def calculate_physics_residual_loss_1d_full(data, c=343.0, rho0=1.21, dt=1e-5, dx=0.032, use_huber=True):
    """
    计算一维声学方程残差 (消声室自由场版 + 终极残差量级配平)
    """
    if data.shape[1] < 3 or data.shape[3] < 3:
        return torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device), \
               torch.tensor(0.0, device=data.device)

    criterion = torch.nn.SmoothL1Loss() if use_huber else torch.nn.MSELoss()
    
    # 🚨 物理学常数：空气特性阻抗
    Z0 = rho0 * c

    # ==========================================
    # 1. 波动方程 (Wave Equation)
    # ==========================================
    psi_center = data[:, 1:-1, :, 1:-1]
    d2psi_dt2 = (data[:, 2:, :, 1:-1] - 2*psi_center + data[:, :-2, :, 1:-1]) / (dt**2)
    d2psi_dx2 = (data[:, 1:-1, :, 2:] - 2*psi_center + data[:, 1:-1, :, :-2]) / (dx**2)

    res_wave_all = (1.0 / (c**2)) * d2psi_dt2 - d2psi_dx2
    
    # 🚨 1：将 Vx, Vy, Vz 的残差乘以 Z0，使其与 P 的残差处于同一量级 (O(1))
    weight_wave = torch.tensor([1.0, Z0, Z0, Z0], device=data.device, dtype=data.dtype).view(1, 1, 4, 1)
    res_wave_all = res_wave_all * weight_wave
    
    loss_wave = criterion(res_wave_all, torch.zeros_like(res_wave_all))

    # ==========================================
    # 提取 P 和 Vx
    # ==========================================
    p = data[:, :, 0:1, :]   
    vx = data[:, :, 1:2, :]  

    # ==========================================
    # 2. 连续性方程 (Continuity Equation)
    # ==========================================
    dp_dt = (p[:, 2:, :, 1:-1] - p[:, :-2, :, 1:-1]) / (2 * dt)
    vx_xp1 = vx[:, 1:-1, :, 2:]
    vx_xm1 = vx[:, 1:-1, :, :-2]
    dvx_dx = (vx_xp1 - vx_xm1) / (2 * dx)

    # 原残差量级为 dv/dx (极小，O(10^-4))
    res_cont = (1.0 / (rho0 * (c**2))) * dp_dt + dvx_dx
    
    # 🚨 核心修复 2：将连续性残差乘以 Z0，拉拔到 O(1) 量级
    res_cont = res_cont * Z0
    loss_cont = criterion(res_cont, torch.zeros_like(res_cont))

    # ==========================================
    # 3. 动量方程 (Euler Equation)
    # ==========================================
    dvx_dt = (vx[:, 2:, :, 1:-1] - vx[:, :-2, :, 1:-1]) / (2 * dt)
    p_xp1 = p[:, 1:-1, :, 2:]
    p_xm1 = p[:, 1:-1, :, :-2]
    dp_dx = (p_xp1 - p_xm1) / (2 * dx)

    # 原残差量级为 dv/dt (由于频率 omega 的存在，已在 O(1) 附近)
    # 所以无需乘 Z0，天然与 P 处于相近量级
    res_euler = dvx_dt + (1.0 / rho0) * dp_dx
    loss_euler = criterion(res_euler, torch.zeros_like(res_euler))

    # ==========================================
    # 4. 无边界条件 (自由场)
    # ==========================================
    loss_bc = torch.tensor(0.0, device=data.device)

    return loss_wave, loss_cont, loss_euler, loss_bc




    


def s3gm_sample_1d_ISS_1dtube_res(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=100, reg_coef_pa = 1e4, reg_coef_ar = 1e4,
                                use_residual_objective=False,
                                data_scalar=None,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True,
                                # --- 物理参数 ---
                                physics_start_step=20,
                                reg_coef_wave=0.1, 
                                reg_coef_cont=1.0, 
                                reg_coef_euler=100,
                                reg_coef_bc=1,
                                normalized_velocity_zero = None,
                                physics_c=343.0,
                                physics_rho0=1.21, 
                                physics_dt=1e-4,
                                physics_dx=1.5/1024,
                                ):

    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_1d_ISS_1dtube_res(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                            alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                            snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                            probability_flow=probability_flow, continuous=continuous,
                                            denoising_steps=denoising_steps, eta=eta, reg_coef_pa = reg_coef_pa, 
                                            # 传入物理参数
                                            data_scalar=data_scalar,
                                            physics_start_step=physics_start_step,
                                            reg_coef_wave=reg_coef_wave,
                                            reg_coef_cont=reg_coef_cont,
                                            reg_coef_euler=reg_coef_euler,
                                            reg_coef_bc = reg_coef_bc,
                                            normalized_velocity_zero=normalized_velocity_zero,
                                            physics_c=physics_c,
                                            physics_rho0=physics_rho0,
                                            physics_dt=physics_dt,
                                            physics_dx=physics_dx
                                            )
    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]

        x_extra, _ = generate_ar_1d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y


def generate_parallel_1d_ISS_1dtube_res(config, net, sde, y, transform, corrector, n_steps=5, 
                              alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                              gamma=1.e-2, num_steps=10,  
                              denoising_steps=100, 
                              overlap=1, eta=0.0, reg_coef_pa = 1e4,
                              device='cpu', dtype='float32', eps=1e-3, 
                              save_sample_path=False, probability_flow=False, 
                              continuous=True, 
                              # --- 物理参数 ---
                              data_scalar=None,
                              physics_start_step=20,
                              reg_coef_wave=0.1, 
                              reg_coef_cont=1.0, 
                              reg_coef_euler=100,
                              reg_coef_bc = 1,
                              normalized_velocity_zero=None,
                              physics_c=343.0,
                              physics_rho0=1.21, 
                              physics_dt=2e-4,
                              physics_dx=1/21,
                              ):
                                      
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))
    ns_real = b * (nf - ol) + ol 
    nc = config.num_conditions
    ds = denoising_steps
    
    # 1D Shape: [Batch, Blocks, Frames, Channels, Space]
    shape = [len(y), b, nf, ncomp + nc, config.image_size] 
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size], 
                                    dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    T_max = sde.T
    eps_min = eps
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:] - x0_hat_temp[:, :, :, :-1])
        reg_loss = lambda_reg * (10 * dx_t.mean() + dx_w.mean())
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), device=x.device)
    
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[],
                  loss_phy=[], loss_wave=[], loss_cont=[], loss_euler=[])
    
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    sde_start_step = ds - 10
    
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next
            
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            # Reshape for 1D: (b n) t c h
            xb = rearrange(x, 'b n t c h -> (b n) t c h')
            
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h -> (b n) t c h')

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    
                    temp_u = x_pred_mean + noise_coeff * noise
                else:
                    with torch.no_grad():
                        f, G = sde.discretize(inp, vec_t)
                        
                        rev_f = f - G[:, None, None, None] ** 2 * score.detach() * 1.0
                        rev_G = torch.zeros_like(G) if probability_flow else G

                        temp_mean = inp - rev_f 
                        temp_u = temp_mean + rev_G[:, None, None, None] * zb
                
                _, std = sde.marginal_prob(xb, vec_t)
                x0_hat = rearrange(std[:, None, None, None] ** 2 * score + inp, '(b n) t c h -> b n t c h', n=b)
                
                # 定义 x0_hat_temp 用于 Loss 计算
                x0_hat_temp = x_to_sample(x0_hat)
                
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())
                
                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                
                # DPS Loss
                loss_dps = torch.sum(((y - transform(x0_hat_temp))**2 / var).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                # Consistency Loss
                loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)
                loss_consis = torch.sum(loss_consis, dim=-1).mean() # Match Base

                loss_consis_para = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, ncomp:]-x0_hat[:, 1:, :ol, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean() # Match Base

                reg_loss = 1 * compute_regularization(x0_hat_temp, reg_coef_pa)
                
                '''
                loss_phy = torch.tensor(0.0, device=x0_hat_temp.device) 
                loss_wave = torch.tensor(0.0)
                loss_cont = torch.tensor(0.0)
                loss_euler = torch.tensor(0.0)
                
                # 只有当系数 > 0 时才计算物理图，否则完全跳过
                run_physics = (reg_coef_wave > 0 or reg_coef_cont > 0 or reg_coef_euler > 0 or reg_coef_bc > 0)
                
                if (ds - i) <= physics_start_step and run_physics:
                    if data_scalar is not None:
                        if ncomp >= 2: 
                            x0_hat_unnorm = x0_hat_temp.clone()
                            x0_hat_unnorm[:, :, :ncomp] = data_scalar(x0_hat_temp[:, :, :ncomp])
                            ''
                            loss_wave, loss_cont, loss_euler, loss_bc = calculate_physics_residual_loss_1d_3(
                                x0_hat_unnorm, 
                                c=physics_c, 
                                rho0=physics_rho0, 
                                dt=physics_dt, 
                                dx=physics_dx
                            )

                            loss_phy = (reg_coef_wave * loss_wave + 
                                        reg_coef_cont * loss_cont + 
                                        reg_coef_euler * loss_euler +
                                        reg_coef_bc * loss_bc)
                        else:
                            if (ds - i) == physics_start_step:
                                print(f"Warning: Physics loss skipped. Expected ncomp >= 2.")
                    else:
                        if (ds - i) == physics_start_step:
                            print("Warning: Physics loss skipped. 'data_scalar' not provided.")
                '''


                loss_phy = torch.tensor(0.0, device=x0_hat_temp.device) 
                loss_wave = torch.tensor(0.0)
                loss_cont = torch.tensor(0.0)
                loss_euler = torch.tensor(0.0)
                loss_bc = torch.tensor(0.0) # 补齐占位
                
                # 🚨 物理触发条件：彻底剔除 reg_coef_bc
                run_physics = (reg_coef_wave > 0 or reg_coef_cont > 0 or reg_coef_euler > 0)
                
                if (ds - i) <= physics_start_step and run_physics:
                    if data_scalar is not None:
                        if ncomp >= 4: 
                            x0_hat_unnorm = x0_hat_temp.clone()
                            x0_hat_unnorm[:, :, :ncomp] = data_scalar(x0_hat_temp[:, :, :ncomp])
                            loss_wave, loss_cont, loss_euler, loss_bc = calculate_physics_residual_loss_1d_full(
                                x0_hat_unnorm, 
                                c=physics_c, 
                                rho0=physics_rho0, 
                                dt=physics_dt, 
                                dx=physics_dx
                            )

                            # 组合最终 Physics Loss (绝对不加 bc)
                            loss_phy = (reg_coef_wave * loss_wave + 
                                        reg_coef_cont * loss_cont + 
                                        reg_coef_euler * loss_euler)
                        else:
                            if (ds - i) == physics_start_step:
                                print(f"Warning: Physics loss skipped. Expected ncomp >= 4 for full vector PDE.")
                    else:
                        if (ds - i) == physics_start_step:
                            print("Warning: Physics loss skipped. 'data_scalar' not provided.")
                # ==========================================

                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para + reg_loss + loss_phy
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_phy'].append(loss_phy.item())
                losses['reg_loss'].append(reg_loss.item())
                
                mode = "ISS" if i < sde_start_step else "SDE"
                
                tqdm_desc = (
                    f'{mode} | T: {loss.item():.2e} | '
                    f'Obs: {alpha * loss_dps.item():.2e} | '
                    f'Phy: {loss_phy.item():.2e} | '
                    f'Reg: {reg_loss.item():.2e}'
                )
                tqdm_setting.set_description(tqdm_desc)
                
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                
                temp = temp_u - dx

            # === 物理硬约束与 Clamp (严格控制开启条件) ===
            if normalized_velocity_zero is not None:
                temp[:, :, 1, -1] = normalized_velocity_zero
                #temp = torch.clamp(temp, min=-100.0, max=100.0) 
            
            temp = temp.detach()

            # 重组样本进入下一步
            x = rearrange(temp, '(b n) t c h -> b n t c h', n=b)

            x_mean_temp = temp_u if i < sde_start_step else temp_mean 
            
            if normalized_velocity_zero is not None:
                x_mean_temp[:, :, 1, -1] = normalized_velocity_zero
                
            x_mean = rearrange(x_mean_temp, '(b n) t c h -> b n t c h', n=b)
                
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses





def calculate_physics_residual_loss_1d_p(data, c=343.0, rho0=1.21, dt=2e-4, dx=1/21, use_huber=True):
    """
    计算一维声学方程残差 (仅声压波动方程)
    修改说明：由于模型没有振速输出，移除了连续性方程、动量方程和速度边界损失。
    """
    # 确保有足够的时间步和空间点
    if data.shape[1] < 3 or data.shape[3] < 3:
        return torch.tensor(0.0, device=data.device)

    # 修改：只取第0个通道 (声压 p), [B, T, 1, H]
    # 注意：这里假设 data 的 shape 是 [B, T, C, H]，且 C=1
    p = data[:, :, 0:1, :] 

    # 定义损失函数
    criterion = torch.nn.SmoothL1Loss() if use_huber else torch.nn.MSELoss()

    # ==========================================
    # 1. 波动方程 (Wave Equation) - 仅依赖 P
    # ==========================================
    p_center = p[:, 1:-1, :, 1:-1]
    
    # 时间二阶导
    d2p_dt2 = (p[:, 2:, :, 1:-1] - 2*p_center + p[:, :-2, :, 1:-1]) / (dt**2)
    # 空间二阶导
    d2p_dx2 = (p[:, 1:-1, :, 2:] - 2*p_center + p[:, 1:-1, :, :-2]) / (dx**2)

    # 计算归一化残差
    res_wave = (1.0 / (c**2)) * d2p_dt2 - d2p_dx2
    loss_wave = criterion(res_wave, torch.zeros_like(res_wave))

    # 修改：只返回波动方程损失
    return loss_wave

def s3gm_sample_1d_ISS_1dtube_res_p(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=100, reg_coef_pa = 1e4, reg_coef_ar = 1e4,
                                use_residual_objective=False,
                                data_scalar=None,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True,
                                # --- 物理参数 ---
                                physics_start_step=20,
                                reg_coef_wave=0.1, 
                                # reg_coef_cont, reg_coef_euler, reg_coef_bc 已移除或不再使用
                                normalized_velocity_zero = None, # 实际上不再起作用，但保留参数位置防止报错
                                physics_c=343.0,
                                physics_rho0=1.21, 
                                physics_dt=1e-4,
                                physics_dx=1.5/1024,
                                ):

    if T_prime_y > 0:
        x_y, _, _ = generate_parallel_1d_ISS_1dtube_res_p(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                                      alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                                      snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                                      probability_flow=probability_flow, continuous=continuous,
                                                      denoising_steps=denoising_steps, eta=eta, reg_coef_pa = reg_coef_pa, 
                                                      # 传入物理参数
                                                      data_scalar=data_scalar,
                                                      physics_start_step=physics_start_step,
                                                      reg_coef_wave=reg_coef_wave,
                                                      # 下面这些参数虽然传入但内部逻辑已移除
                                                      normalized_velocity_zero=normalized_velocity_zero,
                                                      physics_c=physics_c,
                                                      physics_rho0=physics_rho0,
                                                      physics_dt=physics_dt,
                                                      physics_dx=physics_dx
                                                      )
    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]

        x_extra, _ = generate_ar_1d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y


def generate_parallel_1d_ISS_1dtube_res_p(config, net, sde, y, transform, corrector, n_steps=5, 
                              alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                              gamma=1.e-2, num_steps=10,  
                              denoising_steps=100, 
                              overlap=1, eta=0.0, reg_coef_pa = 1e4,
                              device='cpu', dtype='float32', eps=1e-3, 
                              save_sample_path=False, probability_flow=False, 
                              continuous=True, 
                              # --- 物理参数 ---
                              data_scalar=None,
                              physics_start_step=20,
                              reg_coef_wave=0.1, 
                              # 移除不需要的物理系数参数，或者保留但不用
                              reg_coef_cont=0.0, 
                              reg_coef_euler=0.0,
                              reg_coef_bc = 0.0,
                              normalized_velocity_zero=None,
                              physics_c=343.0,
                              physics_rho0=1.21, 
                              physics_dt=2e-4,
                              physics_dx=1/21,
                              ):
                                      
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))
    ns_real = b * (nf - ol) + ol 
    nc = config.num_conditions
    ds = denoising_steps
    
    # 1D Shape: [Batch, Blocks, Frames, Channels, Space]
    shape = [len(y), b, nf, ncomp + nc, config.image_size] 
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size], 
                                    dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    T_max = sde.T
    eps_min = eps
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:] - x0_hat_temp[:, :, :, :-1])
        reg_loss = lambda_reg * (10 * dx_t.mean() + dx_w.mean())
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), device=x.device)
    
    # 修改：移除了 loss_cont, loss_euler
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[],
                  loss_phy=[], loss_wave=[])
    
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    sde_start_step = ds - 10
    
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next
            
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            # Reshape for 1D: (b n) t c h
            xb = rearrange(x, 'b n t c h -> (b n) t c h')
            
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            '''predictor'''
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h -> (b n) t c h')

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    
                    temp_u = x_pred_mean + noise_coeff * noise
                else:
                    with torch.no_grad():
                        f, G = sde.discretize(inp, vec_t)
                        
                        rev_f = f - G[:, None, None, None] ** 2 * score.detach() * 1.0
                        rev_G = torch.zeros_like(G) if probability_flow else G

                        temp_mean = inp - rev_f 
                        temp_u = temp_mean + rev_G[:, None, None, None] * zb
                
                _, std = sde.marginal_prob(xb, vec_t)
                x0_hat = rearrange(std[:, None, None, None] ** 2 * score + inp, '(b n) t c h -> b n t c h', n=b)
                
                # 定义 x0_hat_temp 用于 Loss 计算
                x0_hat_temp = x_to_sample(x0_hat)
                
                if save_sample_path:
                    x0_hats.append(x0_hat.detach().cpu().numpy())
                
                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                
                # DPS Loss
                loss_dps = torch.sum(((y - transform(x0_hat_temp))**2 / var).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                # Consistency Loss
                loss_consis = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat[:, 1:, :ol, :ncomp])**2).reshape(x0_hat.shape[0], x0_hat.shape[1]-1, -1), dim=-1)
                loss_consis = torch.sum(loss_consis, dim=-1).mean() # Match Base

                loss_consis_para = torch.sum(((x0_hat[:, :-1, (nf-ol):nf, ncomp:]-x0_hat[:, 1:, :ol, ncomp:].detach())**2).reshape(x0_hat.shape[0], -1), dim=-1)
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean() # Match Base

                reg_loss = 1 * compute_regularization(x0_hat_temp, reg_coef_pa)
                

                loss_phy = torch.tensor(0.0, device=x0_hat_temp.device) 
                loss_wave = torch.tensor(0.0)
                
                # 修改：只判断 wave 的系数，移除了 cont, euler, bc 的判断
                run_physics = (reg_coef_wave > 0)
                
                if (ds - i) <= physics_start_step and run_physics:
                    if data_scalar is not None:
                        # 修改：将 ncomp >= 2 改为 ncomp >= 1，允许单通道物理计算
                        if ncomp >= 1: 
                            x0_hat_unnorm = x0_hat_temp.clone()
                            # 仅反归一化存在的通道
                            x0_hat_unnorm[:, :, :ncomp] = data_scalar(x0_hat_temp[:, :, :ncomp])
                            
                            # 修改：只接收 wave loss
                            loss_wave = calculate_physics_residual_loss_1d_p(
                                x0_hat_unnorm, 
                                c=physics_c, 
                                rho0=physics_rho0, 
                                dt=physics_dt, 
                                dx=physics_dx
                            )

                            loss_phy = reg_coef_wave * loss_wave
                        else:
                            if (ds - i) == physics_start_step:
                                print(f"Warning: Physics loss skipped. Expected ncomp >= 1.")
                    else:
                        if (ds - i) == physics_start_step:
                            print("Warning: Physics loss skipped. 'data_scalar' not provided.")
                # ==========================================

                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para + reg_loss + loss_phy
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_phy'].append(loss_phy.item())
                losses['reg_loss'].append(reg_loss.item())
                
                mode = "ISS" if i < sde_start_step else "SDE"
                
                tqdm_desc = (
                    f'{mode} | T: {loss.item():.2e} | '
                    f'Obs: {alpha * loss_dps.item():.2e} | '
                    f'Phy: {loss_phy.item():.2e} | '
                    f'Reg: {reg_loss.item():.2e}'
                )
                tqdm_setting.set_description(tqdm_desc)
                
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                
                temp = temp_u - dx

            # === 修改：移除了针对振速通道的硬约束 (Hard BC) ===
            # 因为只有声压通道，没有通道1，这里必须删除
            # if normalized_velocity_zero is not None:
            #     temp[:, :, 1, -1] = normalized_velocity_zero
            
            temp = temp.detach()

            # 重组样本进入下一步
            x = rearrange(temp, '(b n) t c h -> b n t c h', n=b)

            x_mean_temp = temp_u if i < sde_start_step else temp_mean 
            
            # === 修改：移除了针对振速通道的硬约束 ===
            # if normalized_velocity_zero is not None:
            #     x_mean_temp[:, :, 1, -1] = normalized_velocity_zero
                
            x_mean = rearrange(x_mean_temp, '(b n) t c h -> b n t c h', n=b)
                
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses













def s3gm_sample_2d_ISS_kol(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=1000, reg_coef_pa = 1e4, reg_coef_ar = 1e4,
                                use_residual_objective=False,
                                data_scalar=None,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True):

    """
    Performs 2D inpainting and generation using a combination of parallel and
    autoregressive sampling, utilizing generate_parallel_2d_vesde_ddim.
    """
    if T_prime_y > 0:

        x_y, _, _ = generate_parallel_2d_ISS_kol(config, net, sde, y, transform, corrector, n_steps=n_steps,
                                        alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                        snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                        probability_flow=probability_flow, continuous=continuous,
                                        denoising_steps=denoising_steps,eta=eta, reg_coef_pa = reg_coef_pa)
        '''
        x_y, _, _ = generate_parallel_2d(config, net, sde, y, transform, corrector, n_steps=n_steps, 
                                        alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                        snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False, 
                                        probability_flow=probability_flow, continuous=continuous)
        '''
    else:
        x_y = y

    if T_prime > x_y.shape[1]:
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]


        x_extra, _ = generate_ar_2d_ISS_2dsound(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps, reg_coef_ar = reg_coef_ar)
        '''
        x_extra, _ = generate_ar_2d(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps)
        '''
        # Concatenate the generated extra part, excluding the overlapping portion from x_extra
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1)
    else:
        return x_y

def generate_parallel_2d_ISS_kol(config, net, sde, y, transform, corrector, n_steps=5, 
                          alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                          gamma=1.e-2, num_steps=None,  
                          denoising_steps=100, 
                          overlap=1, eta=0.0, reg_coef_pa = 1e4,
                          device='cpu', dtype='float32', eps=1e-3, 
                          save_sample_path=False, probability_flow=False, 
                          continuous=True, data_scalar=None):
    
    dtype_torch = getattr(torch, dtype)
    
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                           sde=sde,
                                           corrector=corrector,
                                           continuous=continuous,
                                           snr=snr,
                                           n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))
    ns_real = b * (nf - ol) + ol 
    nc = config.num_conditions
    ds = denoising_steps
    reg_coef_pa = reg_coef_pa

    shape = [len(y), b, nf, ncomp + nc, config.image_size, config.image_size]
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size, config.image_size], 
                                dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    T_max = sde.T
    eps_min = eps
    
    def adaptive_cosine_timesteps(steps, T, eps=1e-12, offset=0.008):
        s = torch.linspace(0, 1, steps + 1, device=device)
        alpha_bar = torch.cos((s + offset) / (1 + offset) * torch.pi * 0.5) ** 2
        adaptive_t = T * alpha_bar + eps * (1 - alpha_bar)
        adaptive_t, _ = torch.sort(adaptive_t, descending=True)
        adaptive_t[0] = T
        adaptive_t[-1] = eps
        return adaptive_t
    
    #timesteps = adaptive_cosine_timesteps(ds, T_max, eps_min, 0.008)
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        #reg_loss = lambda_reg * dx_t.mean() + 1e2 * (dx_h.mean() + dx_w.mean()) 
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), x.size(5), device=x.device)
    
    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[])
    
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    sde_start_step = ds - 5
    
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            h = t - t_next
            
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')
            
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)

                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1:
                    sigma_current = sigma_current[:, None, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    x_pred = x_pred_mean + noise_coeff * noise

                    
                else:
                    with torch.no_grad():
                        f, G_val = sde.discretize(inp, vec_t)
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0

                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                
                x0_hat_reshaped = rearrange(x0_hat, '(b n) t c h w -> b n t c h w', n=b)
                x0_hat_sample = x_to_sample(x0_hat_reshaped)
                
                var = std_y**2 + gamma * sigma_current**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_sample))**2 / var).reshape(x0_hat_reshaped.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                loss_consis = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, :ncomp].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, :ncomp])**2
                )
                loss_consis = torch.sum(loss_consis, dim=-1).mean()
                
                loss_consis_para = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, ncomp:].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, ncomp:])**2
                )
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                #decay_factor = max(0.1, 1.0 - (i / ds)) 
                reg_loss = 1 * compute_regularization(x0_hat_sample, reg_coef_pa)
                
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para +  reg_loss
                #loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                losses['reg_loss'].append(reg_loss.item())
                
                mode = "ISS" if i < sde_start_step else "SDE"
                tqdm_setting.set_description(f'{mode} | loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | loss consis.: {beta1 * loss_consis.item():.5e} | reg_loss.: { reg_loss.item():.5e}')
                
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8) 
                #momentum = 0.01 * momentum + 0.99 * dx
                #momentum = 0.99 * momentum + 0.01 * dx
                #x_pred = x_pred - momentum
                x_pred = x_pred - dx
            
            x_pred = x_pred.detach()
            #print(x_pred.shape)
            x = rearrange(x_pred, '(b n) t c h w -> b n t c h w', n=b)
            x_mean = rearrange(x_pred if i < sde_start_step else temp_mean,  # SDE没有mean，直接使用x_pred
                              '(b n) t c h w -> b n t c h w', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)

    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses


#鞅性质验证
#########################################################################################################################################
def s3gm_sample_2d_ISS_kol_martingale(config, net, sde, y, transform, corrector, n_steps=5, alpha=1., beta=0.4, gamma=0.5, snr=0.128, std_y=None,
                                T_prime_y=10, T_prime=0, overlap=1,
                                eta=0.0,
                                denoising_steps=1000, reg_coef_pa = 1e4, reg_coef_ar = 1e4,
                                use_residual_objective=False,
                                data_scalar=None,
                                device='cpu', dtype='float32', eps=1e-12,
                                probability_flow=False, continuous=True):
    if T_prime_y > 0:
        # 接收 md_list
        x_y, x_gen, losses, md_list = generate_parallel_2d_ISS_kol_martingale(
                                config, net, sde, y, transform, corrector, n_steps=n_steps,
                                alpha=alpha, beta1=beta, beta2=beta, num_steps=T_prime_y, overlap=overlap,
                                snr=snr, device=device, dtype=dtype, eps=eps, save_sample_path=False,
                                probability_flow=probability_flow, continuous=continuous,
                                denoising_steps=denoising_steps, eta=eta, reg_coef_pa=reg_coef_pa,
                                data_scalar=data_scalar)
    else:
        x_y = y
        md_list = []
        losses = {}

    if T_prime > x_y.shape[1]:
        # AR 部分如果不需要画图，可以不用管
        x0 = x_y[:, -overlap:]
        T_prime_extra = T_prime - x_y.shape[1]
        transform_init = lambda x: x[:, :overlap]
        x_extra, _ = generate_ar_2d_ISS_2dsound(config, net, sde, None, corrector=corrector, shape=None, snr=snr, x0=x0,
                                     n_steps=n_steps, probability_flow=probability_flow,
                                     alpha=gamma, mult=gamma, num_steps=T_prime_extra, overlap=overlap, transform_init=transform_init,
                                     continuous=continuous, device=device, denoise=True, dtype=dtype, eps=eps, reg_coef_ar = reg_coef_ar)
        return np.concatenate([x_y, x_extra[:, overlap:]], axis=1), None, losses, md_list
    else:
        return x_y, None, losses, md_list

def generate_parallel_2d_ISS_kol_martingale(config, net, sde, y, transform, corrector, n_steps=5, 
                          alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                          gamma=1.e-2, num_steps=None,  
                          denoising_steps=100, 
                          overlap=1, eta=0.0, reg_coef_pa = 1e4,
                          device='cpu', dtype='float32', eps=1e-3, 
                          save_sample_path=False, probability_flow=False, 
                          continuous=True, data_scalar=None):
    
    # --- 保持原有初始化逻辑完全不变 ---
    dtype_torch = getattr(torch, dtype)
    
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    ns = num_steps  
    ncomp = config.num_components
    ol = overlap
    b = int(np.ceil((ns - ol) / (nf - ol)))
    ns_real = b * (nf - ol) + ol 
    nc = config.num_conditions
    ds = denoising_steps
    reg_coef_pa = reg_coef_pa # 确保这行和你的一致

    shape = [len(y), b, nf, ncomp + nc, config.image_size, config.image_size]
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size, config.image_size], 
                                dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, :ncomp] = xx[:, i_inv, :, :ncomp]
            sample[:, i_inv * (nf - ol):i_inv * (nf - ol) + nf, ncomp:] = xx[:, i_inv, :, ncomp:]
        return sample

    T_max = sde.T
    eps_min = eps
    
    # 你的代码里直接用的 linspace
    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    eps_t = torch.tensor(eps_min, device=device).float()
    
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_h = torch.abs(x0_hat_temp[:, :, :, :, 1:] - x0_hat_temp[:, :, :, :, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:, :] - x0_hat_temp[:, :, :, :-1, :])
        reg_loss = lambda_reg * (dx_t.mean() + dx_h.mean() + dx_w.mean())
        return reg_loss

    losses = dict(loss=[], loss_dps=[], loss_eq=[], loss_consis=[], loss_consis_para=[], reg_loss=[])
    
    def get_sigmas(t):
        _, std = sde.marginal_prob(None, t)
        return std

    sde_start_step = ds - 5
    
    # --- Martingale 变量 ---
    martingale_diffs = []
    last_x0_hat = None
    # ---------------------

    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h w -> (b n) t c h w')
            
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1: sigma_current = sigma_current[:, None, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                # --- 记录每一步的 Martingale Divergence ---
                if last_x0_hat is not None:
                    diff = torch.norm(x0_hat.detach() - last_x0_hat.detach(), p=2, dim=(1,2,3,4))
                    martingale_diffs.append(diff.mean().item())
                else:
                    # 第一步没有上一步，填 0 或者跳过，为了绘图方便可以填 0
                    martingale_diffs.append(0.0) 
                    
                last_x0_hat = x0_hat.detach().clone()
                # ---------------------------------------------------
                
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1:
                        sigma_next = sigma_next[:, None, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    noise_coeff = torch.sqrt(variance)
                    noise = torch.randn_like(inp)
                    x_pred = x_pred_mean + noise_coeff * noise

                else:
                    with torch.no_grad():
                        f, G_val = sde.discretize(inp, vec_t)
                        # 注意：这里严格保留你的 * 1.0 和 detach
                        rev_f = f - G_val[:, None, None, None, None] ** 2 * score.detach() * 1.0

                        temp_mean = inp - rev_f
                        noise = torch.randn_like(inp)
                        x_pred = temp_mean + G_val[:, None, None, None, None] * noise
                
                x0_hat_reshaped = rearrange(x0_hat, '(b n) t c h w -> b n t c h w', n=b)
                x0_hat_sample = x_to_sample(x0_hat_reshaped)
                
                var = std_y**2 + gamma * sigma_current**2 if std_y is not None else 1.
                loss_dps = torch.sum(((y - transform(x0_hat_sample))**2 / var).reshape(x0_hat_reshaped.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None:
                    loss_dps = loss_dps / 2.
                
                loss_consis = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, :ncomp].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, :ncomp])**2
                )
                loss_consis = torch.sum(loss_consis, dim=-1).mean()
                
                loss_consis_para = torch.sum(
                    (x0_hat_reshaped[:, :-1, (nf - ol):nf, ncomp:].detach() - 
                     x0_hat_reshaped[:, 1:, :ol, ncomp:])**2
                )
                loss_consis_para = torch.sum(loss_consis_para, dim=-1).mean()

                reg_loss = 1 * compute_regularization(x0_hat_sample, reg_coef_pa)
                
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para + reg_loss
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_consis'].append(loss_consis.item())
                losses['loss_consis_para'].append(loss_consis_para.item())
                losses['reg_loss'].append(reg_loss.item())
                
                mode = "ISS" if i < sde_start_step else "SDE"
                
                # 实时显示鞅散度
                curr_md = martingale_diffs[-1] if len(martingale_diffs) > 0 else 0.
                tqdm_setting.set_description(f'{mode} | loss total: {loss.item():.5e} | loss obs.: {alpha * loss_dps.item():.5e} | MD: {curr_md:.4e}')
                
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8) 
                
                x_pred = x_pred - dx
            
            x_pred = x_pred.detach()
            x = rearrange(x_pred, '(b n) t c h w -> b n t c h w', n=b)
            # 注意：这里的逻辑也严格保留
            x_mean = rearrange(x_pred if i < sde_start_step else temp_mean, 
                               '(b n) t c h w -> b n t c h w', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)


    # 直接返回列表 martingale_diffs，而不是均值 --->
    return x_to_sample(x_mean).detach().cpu().numpy(), \
           (x_generated if save_sample_path else None), \
           losses, martingale_diffs
#########################################################################################################################################



def calculate_physics_residual_loss_puvw(data, c=343.0, rho0=1.21, dt=2e-4, dx=1/21, use_huber=True):
    """
    计算 4通道 (p, vx, vy, vz) 的物理残差。
    由于是 1D 线阵列，只能计算 x 方向的导数，因此仅约束 p 和 vx 的耦合关系。
    data shape: [B, T, 4, H]
    """
    # 确保有足够的时间步和空间点
    if data.shape[1] < 3 or data.shape[3] < 3:
        zero = torch.tensor(0.0, device=data.device)
        return zero, zero, zero, zero

    # 提取物理量
    p = data[:, :, 0:1, :]  # Channel 0: Pressure
    vx = data[:, :, 1:2, :] # Channel 1: Axial Velocity (u)
    # vy = data[:, :, 2:3, :] # Channel 2: Transverse v (无法求导，忽略)
    # vz = data[:, :, 3:4, :] # Channel 3: Transverse w (无法求导，忽略)

    # 定义损失函数
    criterion = torch.nn.SmoothL1Loss() if use_huber else torch.nn.MSELoss()

    # ==========================================
    # 1. 轴向 Euler 方程 (Axial Momentum Equation)
    # rho0 * dvx/dt + dp/dx = 0
    # ==========================================
    # 时间导数 dvx/dt
    dvx_dt = (vx[:, 2:, :, 1:-1] - vx[:, :-2, :, 1:-1]) / (2 * dt)
    
    # 空间导数 dp/dx
    p_xp1 = p[:, 1:-1, :, 2:]
    p_xm1 = p[:, 1:-1, :, :-2]
    dp_dx = (p_xp1 - p_xm1) / (2 * dx)

    res_euler = rho0 * dvx_dt + dp_dx
    loss_euler = criterion(res_euler, torch.zeros_like(res_euler))

    # ==========================================
    # 2. 轴向连续性方程近似 (Continuity Equation Approximation)
    # 1/(rho0*c^2) * dp/dt + dvx/dx = 0 (忽略 vy, vz 的散度贡献)
    # ==========================================
    # 时间导数 dp/dt
    dp_dt = (p[:, 2:, :, 1:-1] - p[:, :-2, :, 1:-1]) / (2 * dt)
    
    # 空间导数 dvx/dx
    vx_xp1 = vx[:, 1:-1, :, 2:]
    vx_xm1 = vx[:, 1:-1, :, :-2]
    dvx_dx = (vx_xp1 - vx_xm1) / (2 * dx)

    res_cont = (1.0 / (rho0 * (c**2))) * dp_dt + dvx_dx
    loss_cont = criterion(res_cont, torch.zeros_like(res_cont))

    # ==========================================
    # 3. 1D 波动方程 (Wave Equation on p)
    # 1/c^2 * d2p/dt2 - d2p/dx2 = 0
    # ==========================================
    p_center = p[:, 1:-1, :, 1:-1]
    # 时间二阶导
    d2p_dt2 = (p[:, 2:, :, 1:-1] - 2*p_center + p[:, :-2, :, 1:-1]) / (dt**2)
    # 空间二阶导
    d2p_dx2 = (p[:, 1:-1, :, 2:] - 2*p_center + p[:, 1:-1, :, :-2]) / (dx**2)

    res_wave = (1.0 / (c**2)) * d2p_dt2 - d2p_dx2
    loss_wave = criterion(res_wave, torch.zeros_like(res_wave))

    # ==========================================
    # 4. 边界条件 (例如最右端刚性壁面 vx=0)
    # ==========================================
    vx_outlet = vx[:, :, :, -1] 
    loss_bc = criterion(vx_outlet, torch.zeros_like(vx_outlet))

    return loss_wave, loss_cont, loss_euler, loss_bc




def s3gm_sample_1d_line_array_puvw(config, net, sde, y, transform, corrector, n_steps=5, 
                              alpha=1., beta1=100., beta2=100, snr=0.128, std_y=None, 
                              gamma=1.e-2, num_steps=10,  
                              denoising_steps=100, 
                              overlap=1, eta=0.0, reg_coef_pa = 1e4,
                              device='cpu', dtype='float32', eps=1e-3, 
                              save_sample_path=False, probability_flow=False, 
                              continuous=True, 
                              # --- 物理参数 ---
                              data_scalar=None,
                              physics_start_step=20,
                              reg_coef_wave=0.1, 
                              reg_coef_cont=1.0, 
                              reg_coef_euler=100,
                              reg_coef_bc = 1,
                              normalized_velocity_zero=None,
                              physics_c=343.0,
                              physics_rho0=1.21, 
                              physics_dt=2e-4,
                              physics_dx=1/21,
                              # 新增 T_prime 参数以兼容调用
                              T_prime_y=None, T_prime=None
                              ):
                                      
    dtype_torch = getattr(torch, dtype)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    
    net_fn = lambda a, b: predict_fn(net, sde, a, b, continuous)

    y = torch.from_numpy(y).to(device).type(dtype_torch)

    nf = config.num_frames
    # 兼容传入的 num_steps 或从 T_prime 计算
    if T_prime is not None:
         ns_real = T_prime
         b = int(np.ceil((ns_real - overlap) / (nf - overlap)))
    else:
         ns = num_steps  
         ol = overlap
         b = int(np.ceil((ns - ol) / (nf - ol)))
         ns_real = b * (nf - ol) + ol 

    ncomp = config.num_components
    ol = overlap
    nc = config.num_conditions
    ds = denoising_steps
    
    # Shape: [Batch, Blocks, Frames, Channels, Space]
    shape = [len(y), b, nf, ncomp + nc, config.image_size] 
    
    def x_to_sample(xx, sample=None):
        if sample is None:
            sample = torch.zeros([len(y), ns_real, ncomp + nc, config.image_size], 
                                    dtype=dtype_torch, device=device)
        for i in range(b):
            i_inv = b - i - 1
            # 注意越界检查，最后一块可能比 ns_real 长
            t_start = i_inv * (nf - ol)
            t_end = t_start + nf
            if t_end > ns_real: t_end = ns_real # 简单截断保护
            
            sample[:, t_start:t_end, :ncomp] = xx[:, i_inv, :t_end-t_start, :ncomp]
            sample[:, t_start:t_end, ncomp:] = xx[:, i_inv, :t_end-t_start, ncomp:]
        return sample

    timesteps = torch.linspace(sde.T, eps, sde.N + 1, device=device).float()
    
    x_unknown = sde.prior_sampling(shape).to(device).float()
    x = x_unknown
    x_mean = torch.zeros_like(x)
    x_generated = [x_unknown.detach().cpu().numpy()]
    x0_hats = []

    def compute_regularization(x0_hat_temp, lambda_reg=1e-3):
        dx_t = torch.abs(x0_hat_temp[:, 1:] - x0_hat_temp[:, :-1])
        dx_w = torch.abs(x0_hat_temp[:, :, :, 1:] - x0_hat_temp[:, :, :, :-1])
        reg_loss = lambda_reg * (10 * dx_t.mean() + dx_w.mean())
        return reg_loss

    momentum = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), device=x.device)
    
    losses = dict(loss=[], loss_dps=[], loss_phy=[], reg_loss=[])
    
    sde_start_step = ds - 10
    
    with tqdm(range(ds)) as tqdm_setting:
        for i in range(ds):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            vec_t = torch.ones(shape[0] * b, device=t.device).float() * t
            xb = rearrange(x, 'b n t c h -> (b n) t c h')
            
            # Corrector
            temp, temp_mean = corrector_update_fn(xb, vec_t, net=net)

            # Predictor
            z = torch.randn_like(x)
            zb = rearrange(z, 'b n t c h -> (b n) t c h')

            with torch.enable_grad():
                inp = temp.clone()
                inp.requires_grad_(True)
                score = net_fn(inp, vec_t)
                
                _, sigma_current = sde.marginal_prob(inp, vec_t)
                if sigma_current.dim() == 1: sigma_current = sigma_current[:, None, None, None]
                
                x0_hat = inp + (sigma_current ** 2) * score
                
                # SDE Step (Euler-Maruyama or Ancestral)
                if i < sde_start_step: 
                    vec_t_next = t_next * torch.ones_like(vec_t)
                    _, sigma_next = sde.marginal_prob(None, vec_t_next)
                    if sigma_next.dim() == 1: sigma_next = sigma_next[:, None, None, None]
                    ratio = sigma_next / sigma_current
                    x_pred_mean = ratio * inp + (1 - ratio) * x0_hat
                    
                    variance = eta * (sigma_next**2 * (sigma_current**2 - sigma_next**2)) / sigma_current**2
                    variance = torch.clamp(variance, min=0)
                    temp_u = x_pred_mean + torch.sqrt(variance) * torch.randn_like(inp)
                else:
                    f, G = sde.discretize(inp, vec_t)
                    rev_f = f - G[:, None, None, None] ** 2 * score.detach() * 1.0
                    temp_mean = inp - rev_f 
                    temp_u = temp_mean + G[:, None, None, None] * zb
                
                # --- Loss Calculation ---
                _, std = sde.marginal_prob(xb, vec_t)
                # Reconstruct x0_hat in block format
                x0_hat_block = std[:, None, None, None] ** 2 * score + inp
                x0_hat_full = rearrange(x0_hat_block, '(b n) t c h -> b n t c h', n=b)
                x0_hat_temp = x_to_sample(x0_hat_full)
                
                if save_sample_path: x0_hats.append(x0_hat_block.detach().cpu().numpy())
                
                var = std_y**2 + gamma * std**2 if std_y is not None else 1.
                
                # 1. Measurement Loss (DPS)
                loss_dps = torch.sum(((y - transform(x0_hat_temp))**2 / var).reshape(x0_hat_full.shape[0], -1), dim=-1)
                loss_dps = torch.sum(loss_dps, dim=0)
                if std_y is not None: loss_dps = loss_dps / 2.
                
                # 2. Consistency Loss
                loss_consis = torch.sum(((x0_hat_full[:, :-1, (nf-ol):nf, :ncomp].detach()-x0_hat_full[:, 1:, :ol, :ncomp])**2).reshape(x0_hat_full.shape[0], x0_hat_full.shape[1]-1, -1), dim=-1).mean()
                loss_consis_para = torch.sum(((x0_hat_full[:, :-1, (nf-ol):nf, ncomp:]-x0_hat_full[:, 1:, :ol, ncomp:].detach())**2).reshape(x0_hat_full.shape[0], -1), dim=-1).mean()

                # 3. Regularization Loss
                reg_loss = 1 * compute_regularization(x0_hat_temp, reg_coef_pa)
                
                # 4. Physics Loss (New PUVW version)
                loss_phy = torch.tensor(0.0, device=device) 
                loss_wave = torch.tensor(0.0)
                
                run_physics = (reg_coef_wave > 0 or reg_coef_cont > 0 or reg_coef_euler > 0)
                
                if (ds - i) <= physics_start_step and run_physics:
                    if data_scalar is not None:
                        # 必须有至少2个通道 (p, vx) 才能算物理
                        if ncomp >= 2: 
                            x0_hat_unnorm = x0_hat_temp.clone()
                            # 反归一化
                            x0_hat_unnorm[:, :, :ncomp] = data_scalar(x0_hat_temp[:, :, :ncomp])
                            
                            # 调用新的 PUVW 物理函数
                            loss_wave, loss_cont, loss_euler, loss_bc = calculate_physics_residual_loss_puvw(
                                x0_hat_unnorm, 
                                c=physics_c, 
                                rho0=physics_rho0, 
                                dt=physics_dt, 
                                dx=physics_dx
                            )

                            loss_phy = (reg_coef_wave * loss_wave + 
                                        reg_coef_cont * loss_cont + 
                                        reg_coef_euler * loss_euler +
                                        reg_coef_bc * loss_bc)
                        else:
                             pass # 通道不足
                    else:
                        pass # 缺少 scalar
                
                loss = alpha * loss_dps + beta1 * loss_consis + beta2 * loss_consis_para + reg_loss + loss_phy
                
                losses['loss'].append(loss.item())
                losses['loss_dps'].append(loss_dps.item())
                losses['loss_phy'].append(loss_phy.item())

                mode = "ISS" if i < sde_start_step else "SDE"
                tqdm_setting.set_description(f'{mode}|L:{loss.item():.2e}|Obs:{loss_dps.item():.2e}|Phy:{loss_phy.item():.2e}')
                
                dx = torch.autograd.grad(loss, inp)[0]
                dx = torch.clamp(dx, min=-1e8, max=1e8)
                temp = temp_u - dx

            # BC Hard Constraint (if needed)
            if normalized_velocity_zero is not None and ncomp > 1:
                # 假设 Channel 1 是 vx，在 index -1 处为 0
                temp[:, :, 1, -1] = normalized_velocity_zero
            
            temp = temp.detach()
            x = rearrange(temp, '(b n) t c h -> b n t c h', n=b)
            
            x_mean_temp = temp_u if i < sde_start_step else temp_mean 
            x_mean = rearrange(x_mean_temp, '(b n) t c h -> b n t c h', n=b)
            
            if save_sample_path:
                x_generated.append(x_to_sample(x_mean).detach().cpu().numpy())
            
            tqdm_setting.update(1)
    
    return x_to_sample(x_mean).detach().cpu().numpy(), x0_hats if save_sample_path else None, losses




