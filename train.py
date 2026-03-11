import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from catvton_runtime import (
    DressCodeDataset,
    apply_mask,
    build_condition_input,
    build_models,
    compute_vae_encodings,
    dataset_summary,
    parse_categories,
    resolve_device,
    resolve_weight_dtype,
    save_attention_checkpoint,
    save_preview_grid,
)
from model.utils import get_trainable_module


def parse_args():
    parser = argparse.ArgumentParser(description="Train CatVTON-style attention adapters on DressCode")
    parser.add_argument("--data_root_path", type=str, default="data/DressCode")
    parser.add_argument("--base_model_path", type=str, default="booksforcharlie/stable-diffusion-inpainting")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--project_name", type=str, default="catvton-practice")
    parser.add_argument("--dataset_tag", type=str, default="dresscode-16k-512")
    parser.add_argument("--categories", type=str, default="upper_body,lower_body,dresses")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--validation_batch_size", type=int, default=2)
    parser.add_argument("--num_train_steps", type=int, default=16000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--validation_num_inference_steps", type=int, default=30)
    parser.add_argument("--validation_guidance_scale", type=float, default=2.5)
    parser.add_argument("--condition_dropout_prob", type=float, default=0.1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--resume_attn_ckpt", type=str, default=None)
    parser.add_argument("--resume_attn_version", type=str, default="dresscode-16k-512")
    parser.add_argument("--max_train_pairs_per_category", type=int, default=None)
    parser.add_argument("--max_val_pairs_per_category", type=int, default=None)
    parser.add_argument("--run_validation_at_start", action="store_true")
    return parser.parse_args()


def build_accelerator(args):
    log_with = None if args.report_to == "none" else args.report_to
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    return Accelerator(
        mixed_precision=args.mixed_precision if args.device != "mps" else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=log_with,
        project_config=project_config,
    )


def save_checkpoint(unet, output_dir: str, dataset_tag: str, accelerator: Accelerator):
    checkpoint_dir = Path(output_dir) / dataset_tag / "attention"
    unwrapped_unet = accelerator.unwrap_model(unet)
    attn_modules = get_trainable_module(unwrapped_unet, "attention")
    save_attention_checkpoint(attn_modules, checkpoint_dir)
    return checkpoint_dir


def take_validation_batches(loader):
    return [batch for batch in loader]


def main():
    args = parse_args()
    categories = parse_categories(args.categories)
    requested_device = resolve_device(args.device)
    if requested_device.type != "cuda":
        args.mixed_precision = "no"

    accelerator = build_accelerator(args)
    device = accelerator.device
    weight_dtype = resolve_weight_dtype(device, args.mixed_precision)

    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = DressCodeDataset(
        data_root_path=args.data_root_path,
        categories=categories,
        size=(args.width, args.height),
        split="train",
        max_pairs_per_category=args.max_train_pairs_per_category,
    )
    validation_dataset = DressCodeDataset(
        data_root_path=args.data_root_path,
        categories=categories,
        size=(args.width, args.height),
        split="val",
        max_pairs_per_category=args.max_val_pairs_per_category,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=0,
    )

    vae, unet, _, loaded_from, resolved_base_model_path = build_models(
        base_model_path=args.base_model_path,
        device=device,
        weight_dtype=weight_dtype,
        resume_attn_ckpt=args.resume_attn_ckpt,
        resume_attn_version=args.resume_attn_version,
    )

    for param in unet.parameters():
        param.requires_grad = False
    trainable_modules = get_trainable_module(unet, "attention")
    for param in trainable_modules.parameters():
        param.requires_grad = True

    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    optimizer = torch.optim.AdamW(
        trainable_modules.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )

    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)

    if accelerator.is_main_process and accelerator.log_with is not None:
        accelerator.init_trackers(
            project_name=args.project_name,
            config={
                "dataset": "DressCode",
                "categories": categories,
                "device": str(device),
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_train_steps": args.num_train_steps,
                "resolution": f"{args.height}x{args.width}",
                "resume_attn_ckpt": args.resume_attn_ckpt or "",
                "resume_attn_version": args.resume_attn_version or "",
            },
        )

    if accelerator.is_main_process:
        print(f"device={device} weight_dtype={weight_dtype}")
        print(f"train_summary={dataset_summary(train_dataset)} total={len(train_dataset)}")
        print(f"val_summary={dataset_summary(validation_dataset)} total={len(validation_dataset)}")
        if loaded_from is not None:
            print(f"loaded_attention_checkpoint={loaded_from}")
        print(f"resolved_base_model_path={resolved_base_model_path}")

    if accelerator.is_main_process and args.run_validation_at_start and len(validation_dataset) > 0:
        save_preview_grid(
            unet=accelerator.unwrap_model(unet),
                    vae=vae,
                    device=device,
                    weight_dtype=weight_dtype,
                    base_model_path=resolved_base_model_path,
                    batches=take_validation_batches(validation_loader),
                    output_path=Path(args.output_dir) / "validation" / "step-000000.png",
                    num_inference_steps=args.validation_num_inference_steps,
            guidance_scale=args.validation_guidance_scale,
        )

    global_step = 0
    progress_bar = tqdm(total=args.num_train_steps, disable=not accelerator.is_local_main_process)
    last_loss = None

    while global_step < args.num_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(unet):
                person = batch["person"].to(device, dtype=weight_dtype)
                cloth = batch["cloth"].to(device, dtype=weight_dtype)
                mask = batch["mask"].to(device, dtype=weight_dtype)

                masked_person = apply_mask(person, mask)
                person_latent = compute_vae_encodings(person, vae)
                masked_latent = compute_vae_encodings(masked_person, vae)
                cloth_latent = compute_vae_encodings(cloth, vae)
                mask_latent = F.interpolate(mask, size=person_latent.shape[-2:], mode="nearest")

                concat_dim = -2
                target_latents = torch.cat([person_latent, cloth_latent], dim=concat_dim)
                condition_latents = torch.cat([masked_latent, cloth_latent], dim=concat_dim)
                mask_latents = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

                if args.condition_dropout_prob > 0:
                    drop_mask = torch.rand(target_latents.shape[0], device=target_latents.device) < args.condition_dropout_prob
                    if drop_mask.any():
                        condition_latents = condition_latents.clone()
                        condition_latents[drop_mask, :, condition_latents.shape[-2] // 2 :, :] = 0

                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=target_latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                model_input = build_condition_input(noisy_latents, mask_latents, condition_latents)

                prediction = unet(model_input, timesteps, encoder_hidden_states=None, return_dict=False)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_modules.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            last_loss = float(loss.detach().item())
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{last_loss:.6f}")
            if accelerator.log_with is not None:
                accelerator.log({"train_loss": last_loss}, step=global_step)

            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                checkpoint_dir = save_checkpoint(unet, args.output_dir, args.dataset_tag, accelerator)
                print(f"saved_checkpoint={checkpoint_dir} step={global_step}")

            if accelerator.is_main_process and len(validation_dataset) > 0 and global_step % args.validation_steps == 0:
                save_preview_grid(
                    unet=accelerator.unwrap_model(unet),
                    vae=vae,
                    device=device,
                    weight_dtype=weight_dtype,
                    base_model_path=resolved_base_model_path,
                    batches=take_validation_batches(validation_loader),
                    output_path=Path(args.output_dir) / "validation" / f"step-{global_step:06d}.png",
                    num_inference_steps=args.validation_num_inference_steps,
                    guidance_scale=args.validation_guidance_scale,
                )

            if global_step >= args.num_train_steps:
                break

        if args.save_every_epoch and accelerator.is_main_process:
            save_checkpoint(unet, args.output_dir, args.dataset_tag, accelerator)

    final_checkpoint = None
    if accelerator.is_main_process:
        final_checkpoint = save_checkpoint(unet, args.output_dir, args.dataset_tag, accelerator)
        print(f"final_checkpoint={final_checkpoint}")
        print(f"final_loss={last_loss}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
