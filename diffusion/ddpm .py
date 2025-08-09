import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

class DDPM:
    def __init__(self, model, schedule, image_size=32, device="cuda"):
        self.model = model
        self.image_size = image_size
        self.sched = schedule
        self.device = device

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sched.sqrt_alphas_cumprod[t][:, None, None, None]
        b = self.sched.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return a * x0 + b * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred = self.model(x_noisy, t.float())
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self.sched.betas[t][:, None, None, None]
        one_over_sqrt_alpha = self.sched.one_over_sqrt_alpha[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = self.sched.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        model_mean = one_over_sqrt_alpha * (x - betas_t * self.model(x, t.float()) / sqrt_one_minus_alphas_cumprod)
        if (t == 0).all():
            return model_mean
        noise = torch.randn_like(x)
        var = self.sched.posterior_var[t][:, None, None, None]
        return model_mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n=64):
        self.model.eval()
        x = torch.randn(n, 3, self.image_size, self.image_size, device=self.device)
        for i in reversed(range(self.sched.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
        x = (x.clamp(-1, 1) + 1) * 0.5
        grid = make_grid(x, nrow=int(n**0.5), padding=2)
        return grid

    def save_grid(self, grid, path):
        save_image(grid, path)
