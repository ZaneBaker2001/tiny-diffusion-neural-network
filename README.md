# 🌀 Tiny Diffusion — A Minimal Denoising Diffusion Model

A clean, minimal PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)**, training on CIFAR-10 to generate 32×32 images.  
It’s compact enough to study in a weekend, but powerful enough to produce compelling results.

![Sample images](runs/default/samples_e60.png)

---

## ✨ Features

- **UNet-style architecture** for noise prediction
- **Cosine β-schedule** for smoother training
- **Automatic Mixed Precision (AMP)** for faster training
- **Checkpointing** & sample grids during training
- **From-scratch scheduler** (no HuggingFace dependencies)
- **Gradio web demo** to generate images interactively
- Well-structured repo for easy extension (EMA, CFG, conditioning, etc.)

---

## 📂 Project Structure

```
tiny-diffusion/
├─ README.md
├─ requirements.txt
├─ config.yaml                # Training hyperparameters
├─ data/                      # Auto-downloaded CIFAR-10
├─ models/
│  └─ unet.py                  # UNet noise predictor
├─ diffusion/
│  ├─ scheduler.py             # β-schedule and constants
│  └─ ddpm.py                   # Forward/reverse processes
├─ train.py                    # Main training loop
├─ sample.py                   # Generate from trained model
└─ app.py                      # Gradio interface
```

---

## 🚀 Quickstart

### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train the model
```bash
python train.py
```
> CIFAR-10 will download automatically. Training takes ~1–4 hours on a modern GPU.  
> Samples are saved in `runs/default/` every few epochs.

### 3) Generate samples
```bash
python sample.py
```
Outputs `samples.png` with an image grid.

### 4) Launch the web demo
```bash
python app.py
```
Visit the Gradio link to interactively generate images.

---

## ⚙ Configuration

All training parameters are in `config.yaml`:

```yaml
seed: 1337
device: "cuda"
image_size: 32
channels: 3
dataset: "CIFAR10"
batch_size: 128
steps: 1000
beta_schedule: "cosine"
lr: 2.0e-4
epochs: 60
amp: true
samples_per_grid: 64
out_dir: "runs/default"
```

---

## 📈 Results

After ~60 epochs on CIFAR-10:

| Epoch | Sample Output |
|-------|---------------|
| 10    | ![](runs/default/samples_e10.png) |
| 30    | ![](runs/default/samples_e30.png) |
| 60    | ![](runs/default/samples_e60.png) |

---

## 🔮 Next Steps

- [ ] Add **Exponential Moving Average (EMA)** of weights for crisper samples  
- [ ] Implement **Classifier-Free Guidance** for more controllable outputs  
- [ ] Swap CIFAR-10 with **CelebA-HQ** or custom datasets  
- [ ] Integrate FID/KID score tracking  

---

## 📜 Citation

If you use this code, credit is appreciated:
```bibtex
@misc{tinydiffusion2025,
  author = {Your Name},
  title = {Tiny Diffusion: Minimal Denoising Diffusion Model in PyTorch},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/tiny-diffusion}}
}
```

---

**MIT License** — feel free to use, modify, and learn from this code.  
Made with ❤️ for deep learning enthusiasts.
