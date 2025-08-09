import gradio as gr
import torch, yaml
from models.unet import UNet
from diffusion.scheduler import DiffusionSchedule
from diffusion.ddpm import DDPM

def load(cfg):
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=3, base=64, time_dim=256).to(device)
    model.load_state_dict(torch.load(f"{cfg['out_dir']}/model_e{cfg['epochs']}.pt", map_location=device))
    sched = DiffusionSchedule(cfg["steps"], device=device).build(cfg["beta_schedule"])
    ddpm = DDPM(model, sched, image_size=cfg["image_size"], device=device)
    return ddpm, device

cfg = yaml.safe_load(open("config.yaml"))
ddpm, device = load(cfg)

def generate(n_samples=16):
    with torch.no_grad():
        grid = ddpm.sample(n=int(n_samples))
    # return HWC image
    return grid.permute(1,2,0).cpu().numpy()

with gr.Blocks() as demo:
    gr.Markdown("# Tiny Diffusion — CIFAR-10 Generator")
    ns = gr.Slider(4, 64, value=16, step=4, label="Number of samples")
    btn = gr.Button("Generate")
    img = gr.Image(type="numpy", label="Samples")
    btn.click(fn=generate, inputs=[ns], outputs=[img])

if __name__ == "__main__":
    demo.launch()
