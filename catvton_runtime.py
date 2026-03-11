import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from huggingface_hub import snapshot_download
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter


CATEGORY_MASK_LABELS: Dict[str, List[int]] = {
    "upper_body": [4, 14, 15],
    "lower_body": [5, 6, 8, 12, 13],
    "dresses": [4, 5, 6, 7, 12, 13, 14, 15],
}


@dataclass
class Sample:
    category: str
    person_name: str
    cloth_name: str
    person_path: Path
    cloth_path: Path
    label_map_path: Path
    mask_path: Path


def parse_categories(raw: str) -> List[str]:
    return [category.strip() for category in raw.split(",") if category.strip()]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device)


def resolve_weight_dtype(device: torch.device, mixed_precision: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[mixed_precision]


def resize_and_crop(image: Image.Image, size):
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_w = w
        new_h = w * target_h // target_w
    else:
        new_h = h
        new_w = h * target_w // target_h
    image = image.crop(((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2))
    return image.resize(size, Image.LANCZOS)


def resize_and_padding(image: Image.Image, size):
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB")).transpose(2, 0, 1)
    return torch.from_numpy(array).float() / 127.5 - 1.0


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.array(mask.convert("L"))[None, ...].astype(np.float32) / 255.0
    array[array < 0.5] = 0.0
    array[array >= 0.5] = 1.0
    return torch.from_numpy(array)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def apply_mask(person: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return person * (mask < 0.5)


def compute_vae_encodings(images: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    pixel_values = images.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    return latents * vae.config.scaling_factor


def decode_latents(latents: torch.Tensor, vae: AutoencoderKL, weight_dtype: torch.dtype) -> torch.Tensor:
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents.to(dtype=weight_dtype)).sample
    return (images / 2 + 0.5).clamp(0, 1)


def build_condition_input(
    noisy_latents: torch.Tensor,
    mask_latents: torch.Tensor,
    condition_latents: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([noisy_latents, mask_latents, condition_latents], dim=1)


def make_grid(images: List[Image.Image], cols: int = 4, gap: int = 4) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (cols * width + gap * (cols - 1), rows * height + gap * (rows - 1)), "black")
    for index, image in enumerate(images):
        x = (index % cols) * (width + gap)
        y = (index // cols) * (height + gap)
        canvas.paste(image, (x, y))
    return canvas


def build_agnostic_mask(label_map_path: Path, mask_path: Path, category: str, force: bool = False) -> Image.Image:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.exists() and not force:
        return Image.open(mask_path).convert("L")

    labels = np.array(Image.open(label_map_path).convert("L"))
    mask = np.isin(labels, CATEGORY_MASK_LABELS[category]).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=9))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    mask_img = mask_img.point(lambda x: 255 if x >= 32 else 0, mode="L")
    mask_img.save(mask_path)
    return mask_img


def resolve_attention_checkpoint_dir(attn_ckpt: str, version: Optional[str]) -> Path:
    path = Path(attn_ckpt)
    if path.exists():
        candidates = [path]
        if version:
            candidates.append(path / version / "attention")
        candidates.append(path / "attention")
        for candidate in candidates:
            if candidate.is_dir() and any((candidate / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
                return candidate
        raise FileNotFoundError(f"Could not find attention checkpoint under {attn_ckpt}")

    allow_patterns = [f"{version}/attention/*"] if version else ["*/attention/*", "attention/*"]
    repo_path = Path(snapshot_download(repo_id=attn_ckpt, allow_patterns=allow_patterns))
    return resolve_attention_checkpoint_dir(str(repo_path), version)


def load_attention_checkpoint(trainable_modules: torch.nn.Module, attn_ckpt: str, version: Optional[str]):
    checkpoint_dir = resolve_attention_checkpoint_dir(attn_ckpt, version)
    load_checkpoint_in_model(trainable_modules, checkpoint_dir)
    return checkpoint_dir


def resolve_base_model_path(base_model_path: str) -> str:
    path = Path(base_model_path)
    if path.exists():
        return str(path)

    local_path = snapshot_download(
        repo_id=base_model_path,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "vae/*",
            "unet/*",
        ],
    )
    return str(local_path)


def save_attention_checkpoint(trainable_modules: torch.nn.Module, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = trainable_modules.state_dict()
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    try:
        from safetensors.torch import save_file

        save_file(state_dict, output_dir / "model.safetensors")
    except Exception:
        pass


def build_models(
    base_model_path: str,
    device: torch.device,
    weight_dtype: torch.dtype,
    resume_attn_ckpt: Optional[str] = None,
    resume_attn_version: Optional[str] = None,
):
    resolved_base_model_path = resolve_base_model_path(base_model_path)
    vae = AutoencoderKL.from_pretrained(resolved_base_model_path, subfolder="vae").to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(resolved_base_model_path, subfolder="unet")
    init_adapter(unet, cross_attn_cls=SkipAttnProcessor)
    attn_modules = get_trainable_module(unet, "attention")
    loaded_from = None
    if resume_attn_ckpt:
        loaded_from = load_attention_checkpoint(attn_modules, resume_attn_ckpt, resume_attn_version)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    return vae, unet, attn_modules, loaded_from, resolved_base_model_path


class DressCodeDataset(Dataset):
    def __init__(
        self,
        data_root_path: str,
        categories: Iterable[str],
        size,
        split: str,
        max_pairs_per_category: Optional[int] = None,
    ):
        self.data_root = Path(data_root_path)
        self.categories = list(categories)
        self.size = size
        self.split = split
        self.max_pairs_per_category = max_pairs_per_category
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        split_file = "train_pairs.txt" if self.split == "train" else "test_pairs_paired.txt"
        for category in self.categories:
            category_dir = self.data_root / category
            pair_path = category_dir / split_file
            loaded = 0
            with open(pair_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    person_name, cloth_name = parts[0], parts[1]
                    samples.append(
                        Sample(
                            category=category,
                            person_name=person_name,
                            cloth_name=cloth_name,
                            person_path=category_dir / "images" / person_name,
                            cloth_path=category_dir / "images" / cloth_name,
                            label_map_path=category_dir / "label_maps" / person_name.replace("_0.jpg", "_4.png"),
                            mask_path=category_dir / "agnostic_masks" / person_name.replace(".jpg", ".png"),
                        )
                    )
                    loaded += 1
                    if self.max_pairs_per_category is not None and loaded >= self.max_pairs_per_category:
                        break
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        person = resize_and_crop(Image.open(sample.person_path).convert("RGB"), self.size)
        cloth = resize_and_padding(Image.open(sample.cloth_path).convert("RGB"), self.size)
        mask = resize_and_crop(build_agnostic_mask(sample.label_map_path, sample.mask_path, sample.category), self.size)
        return {
            "category": sample.category,
            "person_name": sample.person_name,
            "cloth_name": sample.cloth_name,
            "person": image_to_tensor(person),
            "cloth": image_to_tensor(cloth),
            "mask": mask_to_tensor(mask),
        }


@torch.no_grad()
def run_tryon_batch(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    device: torch.device,
    weight_dtype: torch.dtype,
    scheduler: DDIMScheduler,
    person: torch.Tensor,
    cloth: torch.Tensor,
    mask: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
):
    person = person.to(device, dtype=weight_dtype)
    cloth = cloth.to(device, dtype=weight_dtype)
    mask = mask.to(device, dtype=weight_dtype)

    masked_person = apply_mask(person, mask)
    masked_latent = compute_vae_encodings(masked_person, vae)
    cloth_latent = compute_vae_encodings(cloth, vae)
    mask_latent = F.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")

    concat_dim = -2
    masked_latent_concat = torch.cat([masked_latent, cloth_latent], dim=concat_dim)
    mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
    latents = torch.randn_like(masked_latent_concat)

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    cond = masked_latent_concat
    uncond = torch.cat([masked_latent, torch.zeros_like(cloth_latent)], dim=concat_dim)
    cond = torch.cat([uncond, cond], dim=0)
    mask_latent_concat = torch.cat([mask_latent_concat, mask_latent_concat], dim=0)

    for timestep in scheduler.timesteps:
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        unet_input = torch.cat([latent_model_input, mask_latent_concat, cond], dim=1)
        noise_pred = unet(unet_input, timestep, encoder_hidden_states=None, return_dict=False)[0]
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    latents = latents.split(latents.shape[-2] // 2, dim=-2)[0]
    results = decode_latents(latents, vae, weight_dtype)
    return results, masked_person


@torch.no_grad()
def save_preview_grid(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    device: torch.device,
    weight_dtype: torch.dtype,
    base_model_path: str,
    batches,
    output_path: Path,
    num_inference_steps: int,
    guidance_scale: float,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unet.eval()
    scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")
    all_images: List[Image.Image] = []
    for batch in batches:
        results, masked_person = run_tryon_batch(
            unet=unet,
            vae=vae,
            device=device,
            weight_dtype=weight_dtype,
            scheduler=scheduler,
            person=batch["person"],
            cloth=batch["cloth"],
            mask=batch["mask"],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        person = batch["person"]
        cloth = batch["cloth"]
        for idx in range(results.shape[0]):
            all_images.extend(
                [
                    tensor_to_pil((person[idx].float() + 1) / 2),
                    tensor_to_pil((masked_person[idx].float() + 1) / 2),
                    tensor_to_pil((cloth[idx].float() + 1) / 2),
                    tensor_to_pil(results[idx]),
                ]
            )
    make_grid(all_images, cols=4).save(output_path)
    unet.train()


def dataset_summary(dataset: DressCodeDataset) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in dataset.samples:
        counts[sample.category] = counts.get(sample.category, 0) + 1
    return counts
