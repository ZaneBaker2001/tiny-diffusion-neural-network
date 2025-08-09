import torch
import numpy as np

def cosine_beta_schedule(T, s=0.008):
    # https://arxiv.org/abs/2102.09672
    steps = T + 1
    x = np.linspace(0, T, steps)
    alphas_cumprod = np.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)

class DiffusionSchedule:
    def __init__(self, T=1000, device="cuda"):
        self.T = T
        self.device = device

    def build(self, schedule="cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        else:
            raise NotImplementedError(schedule)
        betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.one_over_sqrt_alpha = torch.sqrt(1.0 / alphas)
        self.posterior_var = betas * (1 - alphas_cumprod.roll(1, 0)) / (1 - alphas_cumprod)
        self.posterior_var[0] = betas[0]
        return self
