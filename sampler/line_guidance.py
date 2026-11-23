"""Differentiable known-tone and along-line Euler constraints in [p, Z0*v] units."""
import math
import torch


class LineGuidance:
    def __init__(self, samples, frequencies, device='cpu', dtype=torch.float32,
                 fs=100000., dx=-.032, c=343., euler_weight=0., tone_weight=0.):
        assert samples > 8 and dx != 0 and all(0 < f < fs/2 for f in frequencies)
        assert euler_weight >= 0 and tone_weight >= 0
        t = torch.arange(samples, dtype=torch.float64) / fs
        columns = [v for f in frequencies for v in
                   (torch.cos(2*math.pi*f*t), torch.sin(2*math.pi*f*t))]
        columns += [torch.ones_like(t), t-t.mean()]
        design = torch.stack(columns, dim=1)
        assert torch.linalg.matrix_rank(design).item() == design.shape[1]
        self.pinv = torch.linalg.pinv(design).to(device=device, dtype=dtype)
        self.tones = design[:, :2*len(frequencies)].to(device=device, dtype=dtype)
        self.k = (2*math.pi*torch.as_tensor(frequencies, dtype=dtype, device=device)/c)
        self.samples, self.dx = samples, dx
        self.euler_weight, self.tone_weight = euler_weight, tone_weight

    def coefficients(self, x):
        """Input B,T,4,X; output B,F,4,X complex pressure-scale coefficients."""
        assert x.ndim == 4 and x.shape[1] == self.samples and x.shape[2] == 4
        coefficient = torch.einsum('qt,btcs->bqcs', self.pinv, x)
        pairs = coefficient[:, :len(self.k)*2]
        return torch.complex(pairs[:, 0::2], -pairs[:, 1::2]), pairs

    def project(self, x):
        _, pairs = self.coefficients(x)
        return torch.einsum('tq,bqcs->btcs', self.tones, pairs)

    def terms(self, x):
        assert x.shape[-1] >= 7
        coeff, pairs = self.coefficients(x)
        p, ux = coeff[:, :, 0], coeff[:, :, 1]
        # A signed grid spacing preserves the physical x derivative on descending arrays.
        dp = (-p[..., :-6] + 9*p[..., 1:-5] - 45*p[..., 2:-4]
              + 45*p[..., 4:-2] - 9*p[..., 5:-1] + p[..., 6:])/(60*self.dx)
        residual = ux[..., 3:-3] + dp/(1j*self.k[None, :, None])
        # N/2 maps complex sinusoidal amplitude energy to approximate time-domain SSE.
        euler = .5*self.samples*residual.abs().square().sum()
        projected = torch.einsum('tq,bqcs->btcs', self.tones, pairs)
        tone = (x-projected).square().sum()
        return euler, tone

    def __call__(self, x):
        euler, tone = self.terms(x)
        return self.euler_weight*euler + self.tone_weight*tone


def join_windows(x, overlap):
    """Keep the earlier block's values in each overlapping time segment."""
    return torch.cat((x[:, 0], x[:, 1:, overlap:].reshape(x.shape[0], -1, *x.shape[3:])), dim=1)


def sample_line_array(config, net, sde, observation, sensor_indices, frequencies,
                      settings, device, seed):
    """Sample one four-channel line field with the Figure 5 update sequence."""
    import numpy as np
    from einops import rearrange
    from sampler import setup_seed
    from trainer.loss import predict_fn

    if (config.num_components, config.num_conditions, config.image_size) != (4, 0, 32):
        raise ValueError("Figure 5 requires four components, no conditions and 32 positions")
    if any(settings[key] != 0 for key in ('wave_weight', 'continuity_weight', 'euler_weight')):
        raise ValueError("Figure 5 uses zero along-line physics weights")
    setup_seed(seed)
    y = torch.as_tensor(observation, dtype=torch.float32, device=device)
    if y.ndim != 4 or y.shape[0] != 1 or y.shape[2] != 4:
        raise ValueError("observation must have shape 1,time,4,input_positions")
    frames, overlap = config.num_frames, settings['overlap']
    samples, steps = y.shape[1], settings['reverse_steps']
    blocks = int(np.ceil((samples-overlap)/(frames-overlap)))
    if blocks*(frames-overlap)+overlap != samples or steps != sde.N:
        raise ValueError("time-window or reverse-step count does not match the sampler")
    indices = torch.as_tensor(sensor_indices, device=device)
    transform = lambda value: torch.index_select(value[:, :samples, :4], -1, indices)
    shape = [1, blocks, frames, 4, config.image_size]
    x = sde.prior_sampling(shape).to(device).float()
    timesteps = torch.linspace(sde.T, settings['eps'], sde.N+1, device=device).float()
    guidance = LineGuidance(samples, frequencies, device=device,
                            euler_weight=0., tone_weight=settings['tone_weight'])
    losses = []
    for i in range(steps):
        t, t_next = timesteps[i], timesteps[i+1]
        vec_t = torch.ones(blocks, device=device).float()*t
        xb = rearrange(x, 'b n t c h -> (b n) t c h')
        # Both random draws are retained when eta is zero to preserve each seed's stream.
        z = torch.randn_like(x)
        zb = rearrange(z, 'b n t c h -> (b n) t c h')
        with torch.enable_grad():
            inp = xb.clone().requires_grad_(True)
            score = predict_fn(net, sde, inp, vec_t, continuous=True)
            _, sigma_current = sde.marginal_prob(inp, vec_t)
            sigma_current = sigma_current[:, None, None, None]
            x0_hat = inp+sigma_current**2*score
            if i < steps-settings['stochastic_tail_steps']:
                _, sigma_next = sde.marginal_prob(None, t_next*torch.ones_like(vec_t))
                sigma_next = sigma_next[:, None, None, None]
                ratio = sigma_next/sigma_current
                x_pred_mean = ratio*inp+(1-ratio)*x0_hat
                variance = settings['eta']*(sigma_next**2*(sigma_current**2-sigma_next**2))/sigma_current**2
                noise = torch.randn_like(inp)
                temp_u = x_pred_mean+torch.sqrt(torch.clamp(variance, min=0))*noise
            else:
                with torch.no_grad():
                    drift, diffusion = sde.discretize(inp, vec_t)
                    reverse_drift = drift-diffusion[:, None, None, None]**2*score.detach()
                    temp_mean = inp-reverse_drift
                    temp_u = temp_mean+diffusion[:, None, None, None]*zb
            _, std = sde.marginal_prob(xb, vec_t)
            x0_blocks = rearrange(std[:, None, None, None]**2*score+inp,
                                  '(b n) t c h -> b n t c h', n=blocks)
            denoised = join_windows(x0_blocks, overlap)
            observation_loss = torch.sum(torch.sum(((y-transform(denoised))**2).reshape(1, -1), dim=-1), dim=0)
            consistency_difference = (
                x0_blocks[:, :-1, frames-overlap:frames].detach()-
                x0_blocks[:, 1:, :overlap])**2
            consistency = torch.sum(torch.sum(consistency_difference.reshape(
                1, blocks-1, -1), dim=-1), dim=-1).mean()
            dt = torch.abs(denoised[:, 1:]-denoised[:, :-1])
            dx = torch.abs(denoised[:, :, :, 1:]-denoised[:, :, :, :-1])
            variation = settings['variation_weight']*(10*dt.mean()+dx.mean())
            tone = guidance(denoised) if i >= settings['tone_start_step'] else denoised.new_zeros(())
            loss = (settings['observation_weight']*observation_loss+
                    settings['overlap_weight']*consistency+variation+tone)
            gradient = torch.clamp(torch.autograd.grad(loss, inp)[0], min=-1e8, max=1e8)
            multiplier = settings['early_guidance_scale'] if i < settings['early_guidance_steps'] else 1.
            temp = (temp_u-multiplier*gradient).detach()
        x = rearrange(temp, '(b n) t c h -> b n t c h', n=blocks)
        losses.append(dict(step=i, total=loss.item(), observation=observation_loss.item(),
                           overlap=consistency.item(), variation=variation.item(), tones=tone.item()))
    return denoised.detach().cpu().numpy(), losses
