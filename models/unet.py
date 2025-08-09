import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Tiny building blocks ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device) * -torch.log(torch.tensor(10000.0, device=device)) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def conv3x3(in_c, out_c, stride=1, groups=1):
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, groups=groups)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = conv3x3(in_c, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_c)
        )
        self.res = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        # inject time
        time = self.time_mlp(t)[:, :, None, None]
        h = h + time
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.res(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out, time_dim):
        super().__init__()
        self.block1 = ResBlock(c_in, c_out, time_dim)
        self.block2 = ResBlock(c_out, c_out, time_dim)
        self.down = nn.Conv2d(c_out, c_out, 4, stride=2, padding=1)
    def forward(self, x, t):
        x = self.block1(x, t); x = self.block2(x, t)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, c_in, c_out, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in, 4, stride=2, padding=1)
        self.block1 = ResBlock(c_in + c_out, c_out, time_dim)
        self.block2 = ResBlock(c_out, c_out, time_dim)
    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t); x = self.block2(x, t)
        return x

# --- UNet ---
class UNet(nn.Module):
    def __init__(self, in_ch=3, base=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )
        self.in_conv = conv3x3(in_ch, base)

        self.down1 = Down(base, base*2, time_dim)
        self.down2 = Down(base*2, base*4, time_dim)
        self.mid1  = ResBlock(base*4, base*4, time_dim)
        self.mid2  = ResBlock(base*4, base*4, time_dim)
        self.up2   = Up(base*4, base*2, time_dim)
        self.up1   = Up(base*2, base,   time_dim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x0 = self.in_conv(x)
        x1, s1 = self.down1(x0, t)
        x2, s2 = self.down2(x1, t)
        m = self.mid2(self.mid1(x2, t), t)
        u2 = self.up2(m, s2, t)
        u1 = self.up1(u2, s1, t)
        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out
