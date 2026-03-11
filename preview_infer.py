import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from catvton_runtime import (
    DressCodeDataset,
    build_models,
    dataset_summary,
    parse_categories,
    resolve_device,
    resolve_weight_dtype,
    save_preview_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed paired validation previews from an attention checkpoint.")
    parser.add_argument("--data_root_path", type=str, default="data/DressCode")
    parser.add_argument("--base_model_path", type=str, default="booksforcharlie/stable-diffusion-inpainting")
    parser.add_argument("--resume_attn_ckpt", type=str, required=True)
    parser.add_argument("--resume_attn_version", type=str, default="dresscode-16k-512")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--categories", type=str, default="upper_body,lower_body,dresses")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--validation_batch_size", type=int, default=2)
    parser.add_argument("--max_val_pairs_per_category", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    return parser.parse_args()


def main():
    args = parse_args()
    categories = parse_categories(args.categories)
    device = resolve_device(args.device)
    weight_dtype = resolve_weight_dtype(device, args.mixed_precision)

    dataset = DressCodeDataset(
        data_root_path=args.data_root_path,
        categories=categories,
        size=(args.width, args.height),
        split="val",
        max_pairs_per_category=args.max_val_pairs_per_category,
    )
    print(f"device={device} weight_dtype={weight_dtype}")
    print(f"val_summary={dataset_summary(dataset)} total={len(dataset)}")

    loader = DataLoader(dataset, batch_size=args.validation_batch_size, shuffle=False, num_workers=0)
    vae, unet, _, loaded_from, resolved_base_model_path = build_models(
        base_model_path=args.base_model_path,
        device=device,
        weight_dtype=weight_dtype,
        resume_attn_ckpt=args.resume_attn_ckpt,
        resume_attn_version=args.resume_attn_version,
    )
    print(f"loaded_attention_checkpoint={loaded_from}")
    print(f"resolved_base_model_path={resolved_base_model_path}")

    save_preview_grid(
        unet=unet,
        vae=vae,
        device=device,
        weight_dtype=weight_dtype,
        base_model_path=resolved_base_model_path,
        batches=[batch for batch in loader],
        output_path=Path(args.output_path),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    print(f"saved_preview={args.output_path}")


if __name__ == "__main__":
    main()
