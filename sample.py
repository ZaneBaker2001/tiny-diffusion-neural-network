import torch, yaml
from models.unet import UNet
from diffusion.scheduler import DiffusionSchedule
from diffusion.ddpm import DDPM
from torchvision.utils import save_image

def load_model(path, device):
    model = UNet(in_ch=3, base=64, time_dim=256).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    model = load_model(f"{cfg['out_dir']}/model_e{cfg['epochs']}.pt", device)
    sched = DiffusionSchedule(cfg["steps"], device=device).build(cfg["beta_schedule"])
    ddpm = DDPM(model, sched, image_size=cfg["image_size"], device=device)
    grid = ddpm.sample(n=64)
    save_image(grid, "samples.png")
    print("Saved to samples.png")
