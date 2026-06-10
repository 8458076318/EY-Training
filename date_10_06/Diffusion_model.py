# -*- coding: utf-8 -*-
"""
This script keeps the notebook's lesson flow but makes the runtime behavior
safe for a normal local environment:
- it falls back to a synthetic image if CIFAR-10 is unavailable
- it runs pretrained Stable Diffusion demos in CPU-safe mode when possible
- it skips pretrained Stable Diffusion demos only when `diffusers` or model
  weights are not present
- it saves figures to disk instead of relying on an interactive notebook UI
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import torchvision
    import torchvision.transforms as transforms
except Exception as exc:  # pragma: no cover - optional dependency guard
    torchvision = None
    transforms = None
    TORCHVISION_IMPORT_ERROR = exc
else:
    TORCHVISION_IMPORT_ERROR = None

try:
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
except Exception as exc:  # pragma: no cover - optional dependency guard
    StableDiffusionImg2ImgPipeline = None
    StableDiffusionPipeline = None
    DIFFUSERS_IMPORT_ERROR = exc
else:
    DIFFUSERS_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_IMAGE = BASE_DIR / "output.png"
EDITED_IMAGE = BASE_DIR / "edited.png"
EDITED_OUTPUT_IMAGE = BASE_DIR / "edited_2.png"
NOISE_FIGURE = BASE_DIR / "diffusion_fixed_beta.png"
FORWARD_FIGURE = BASE_DIR / "diffusion_forward_process.png"


# Mathematical models related to diffusion
# First we will take an image from Dataset
# Observe the noising step (beta which decides the rate of noising) of this image
# Denoise the same image (train our model)
# Trained model to decipher a sanskrit manuscript (diffusers library)


# Mathematical models related to diffusion
#
# Forward diffusion equation
# x(t) = (sqrt)(1-beta)x(t-1) + sqrt(beta*epsilon)
#
# x(t) = Noisy image
# epsilon = Gaussian noise
# beta = noise schedule


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor in [0, 1] to a PIL image."""

    arr = x.detach().cpu().clamp(0, 1)
    arr = (arr * 255).to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


def create_synthetic_demo_image(size: int = 32) -> tuple[torch.Tensor, str]:
    """Create a deterministic fallback image when CIFAR-10 is unavailable."""

    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, size),
        torch.linspace(0, 1, size),
        indexing="ij",
    )
    red = xx
    green = yy
    blue = 0.5 * (torch.sin(4 * math.pi * xx) * torch.cos(4 * math.pi * yy) + 1.0)
    x0 = torch.stack([red, green, blue], dim=0).to(torch.float32)
    return x0, "synthetic_gradient"


def load_demo_image() -> tuple[torch.Tensor, str, str]:
    """Load the first CIFAR-10 image or fall back to a synthetic tensor."""

    if torchvision is None or transforms is None:
        x0, label_name = create_synthetic_demo_image()
        return x0, label_name, f"fallback: torchvision unavailable ({TORCHVISION_IMPORT_ERROR})"

    transform = transforms.Compose([transforms.ToTensor()])
    last_error: Exception | None = None

    for download in (False, True):
        try:
            dataset = torchvision.datasets.CIFAR10(
                root=str(DATA_DIR),
                train=True,
                download=download,
                transform=transform,
            )
            x0, label = dataset[0]
            return x0, dataset.classes[label], f"CIFAR10(download={download})"
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc

    x0, label_name = create_synthetic_demo_image()
    return x0, label_name, f"fallback synthetic image ({last_error})"


def add_noise(x: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    """Apply the notebook's one-step forward diffusion approximation."""

    beta_tensor = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
    noise = torch.randn_like(x)
    return torch.sqrt(1 - beta_tensor) * x + torch.sqrt(beta_tensor) * noise


def save_current_figure(path: Path, *, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fixed_beta_noise(x0: torch.Tensor) -> None:
    """Visualize how larger beta values destroy the image."""

    betas = [0.01, 0.05, 0.1, 0.3, 0.6]
    plt.figure(figsize=(15, 3))

    plt.subplot(1, len(betas) + 1, 1)
    plt.imshow(x0.permute(1, 2, 0).clamp(0, 1))
    plt.title("x0")
    plt.axis("off")

    for i, beta in enumerate(betas):
        xt = add_noise(x0, beta)
        plt.subplot(1, len(betas) + 1, i + 2)
        plt.imshow(xt.permute(1, 2, 0).clamp(0, 1))
        plt.title(f"beta={beta}")
        plt.axis("off")

    plt.suptitle("Forward diffusion with a fixed beta", y=1.02)
    save_current_figure(NOISE_FIGURE)


def build_noise_schedule(num_timesteps: int = 1000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_cumprod


def forward_diffusion_at_t(x0: torch.Tensor, t: int, alpha_cumprod: torch.Tensor) -> torch.Tensor:
    """Apply forward diffusion to `x0` at timestep `t`."""

    if t < 1 or t > len(alpha_cumprod):
        raise ValueError(f"t must be in [1, {len(alpha_cumprod)}], got {t}")

    noise = torch.randn_like(x0)
    idx = t - 1
    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod[idx])
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod[idx])
    return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise


def plot_forward_diffusion(x0: torch.Tensor, alpha_cumprod: torch.Tensor) -> None:
    visualization_timesteps = [1, 50, 100, 200, 500, 999]

    plt.figure(figsize=(15, 4))
    plt.subplot(1, len(visualization_timesteps) + 1, 1)
    plt.imshow(x0.permute(1, 2, 0).clamp(0, 1))
    plt.title("Original x0")
    plt.axis("off")

    for i, t in enumerate(visualization_timesteps):
        xt = forward_diffusion_at_t(x0, t, alpha_cumprod)
        plt.subplot(1, len(visualization_timesteps) + 1, i + 2)
        plt.imshow(xt.permute(1, 2, 0).clamp(0, 1))
        plt.title(f"t={t}")
        plt.axis("off")

    plt.suptitle("Forward diffusion over timesteps", y=1.02)
    save_current_figure(FORWARD_FIGURE)


def create_placeholder_edit_source(x0: torch.Tensor) -> Image.Image:
    """Ensure the image-to-image demo has a local source image."""

    if EDITED_IMAGE.exists():
        return Image.open(EDITED_IMAGE).convert("RGB")

    source = tensor_to_pil(x0)
    draw = ImageDraw.Draw(source)
    draw.rectangle((2, 2, 29, 29), outline=(255, 255, 255), width=1)
    draw.text((4, 4), "edit", fill=(255, 255, 255))
    source.save(EDITED_IMAGE)
    return source


def load_diffusion_pipeline(pipeline_kind: str):
    """Load a Stable Diffusion pipeline in CPU-safe mode if possible."""

    if StableDiffusionPipeline is None or StableDiffusionImg2ImgPipeline is None:
        print(f"[skip] {pipeline_kind}: diffusers is unavailable ({DIFFUSERS_IMPORT_ERROR})")
        return None

    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline_cls = (
        StableDiffusionPipeline if pipeline_kind == "txt2img" else StableDiffusionImg2ImgPipeline
    )

    try:
        pipe = pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            local_files_only=False,
        ).to("cpu")
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        return pipe
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[skip] {pipeline_kind}: could not load cached model -> {exc}")
        return None


def run_pretrained_text_to_image() -> None:
    """Apply a pretrained diffusion model when the dependencies are available."""

    pipe = load_diffusion_pipeline("txt2img")
    if pipe is None:
        return

    print("Running pretrained text-to-image demo on CPU.")
    prompt = "English alphabet"
    image = pipe(prompt, num_inference_steps=20).images[0]
    image.save(OUTPUT_IMAGE)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    save_current_figure(BASE_DIR / "stable_diffusion_text.png", tight=False)


def run_pretrained_image_to_image(x0: torch.Tensor) -> None:
    """Use the correct image-to-image pipeline when cached weights exist."""

    pipe = load_diffusion_pipeline("img2img")
    if pipe is None:
        return

    init_image = create_placeholder_edit_source(x0)
    prompt = "Can you increase the resolution, sharpen facial features"
    result = pipe(prompt=prompt, image=init_image, strength=0.7, num_inference_steps=20).images[0]
    result.save(EDITED_OUTPUT_IMAGE)
    plt.figure(figsize=(6, 6))
    plt.imshow(result)
    plt.axis("off")
    save_current_figure(BASE_DIR / "stable_diffusion_img2img.png", tight=False)


# Difference between DDPM vs DDIM
# DDPM = Probabilistic Denoising model (stochastic) 1000 -> 995 -> 985 -> 965 -> 930 ->.....->0
# DDIM = Implicit Denoising model (deterministic) 1000 -> 800 -> 600 -> 400 -> 200 -> ....->0
# Sampling procedure is different


def main() -> None:
    print("Loading demo image...")
    x0, label_name, source = load_demo_image()
    print(f"Class label/source: {label_name} ({source})")

    print("Saving fixed-beta diffusion visualization...")
    plot_fixed_beta_noise(x0)
    print(f"Saved -> {NOISE_FIGURE}")

    print("Building linear noise schedule...")
    betas, alphas, alpha_cumprod = build_noise_schedule()
    print(f"Number of timesteps (T): {len(betas)}")
    print(f"First 5 betas: {betas[:5]}")
    print(f"Last 5 betas: {betas[-5:]}")

    print("Saving forward diffusion visualization...")
    plot_forward_diffusion(x0, alpha_cumprod)
    print(f"Saved -> {FORWARD_FIGURE}")

    print("Trying pretrained diffusion demos...")
    run_pretrained_text_to_image()
    run_pretrained_image_to_image(x0)

    print("Done.")
    if OUTPUT_IMAGE.exists():
        print(f"Text-to-image output -> {OUTPUT_IMAGE}")
    if EDITED_OUTPUT_IMAGE.exists():
        print(f"Image-to-image output -> {EDITED_OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
