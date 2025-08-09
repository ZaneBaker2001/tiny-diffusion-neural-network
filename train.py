import os, random
import torch, yaml
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
from tqdm import tqdm
from models.unet import UNet
from diffusion.scheduler import DiffusionSchedule
from diffusion.ddpm import DDPM

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], pin_memory=True)

    model = UNet(in_ch=3, base=64, time_dim=256).to(device)
    sched = DiffusionSchedule(cfg["steps"], device=device).build(cfg["beta_schedule"])
    ddpm = DDPM(model, sched, image_size=cfg["image_size"], device=device)
    opt = AdamW(model.parameters(), lr=cfg["lr"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])
    global_step = 0
    os.makedirs(cfg["out_dir"], exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, sched.T, (x.size(0),), device=device, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=cfg["amp"]):
                loss = ddpm.p_losses(x, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            if global_step % cfg["log_every"] == 0:
                pbar.set_postfix(loss=float(loss))

        # sample a grid every few epochs
        if epoch % cfg["ckpt_every"] == 0:
            torch.save(model.state_dict(), f"{cfg['out_dir']}/model_e{epoch}.pt")
            with torch.no_grad():
                grid = ddpm.sample(n=cfg["samples_per_grid"])
                ddpm.save_grid(grid, f"{cfg['out_dir']}/samples_e{epoch}.png")

if __name__ == "__main__":
    main()
