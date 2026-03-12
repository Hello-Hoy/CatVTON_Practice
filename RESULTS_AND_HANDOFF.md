# CatVTON Full Run Results and Handoff

## What finished

- Final full run completed on March 13, 2026 in `outputs/full-resume`.
- Final exact-resume metadata reports `global_step = 16000`.
- Final reusable attention checkpoint:
  - `outputs/full-resume/dresscode-16k-512/attention/pytorch_model.bin`
- Final validation previews:
  - `outputs/full-resume/validation/post-train-paired-preview.png`
  - `outputs/full-resume/validation/post-train-unpaired-preview.png`

## Training setup

- Dataset: local DressCode under `data/DressCode`
- Categories: `upper_body`, `lower_body`, `dresses`
- Base model: `booksforcharlie/stable-diffusion-inpainting`
- VAE: `stabilityai/sd-vae-ft-mse`
- Fine-tuning target: CatVTON-style attention-only adapters
- Device used for final run: single RTX 4080 SUPER
- Main training command shape:
  - `train_batch_size=2`
  - `gradient_accumulation_steps=8`
  - `num_workers=8`
  - `num_train_steps=16000`
  - `checkpointing_steps=500`
  - `validation_steps=20000` during the final long run to avoid mid-run validation stalls

## How to read the results

### Paired preview

The paired preview is the main sanity check because the garment already belongs to the target person/outfit pair. The final grid shows that the checkpoint can usually reconstruct the original garment category, silhouette, and rough texture placement without identity collapse. Tops, pants, and many dresses align well with the source pose.

The remaining weak points are mostly around mask quality and fine detail:

- light-colored dresses lose some edge detail
- long hems and sleeves can look softer than the ground truth
- some lower-body transfers are slightly over-smoothed

This matches the current local mask pipeline, which derives agnostic masks from `label_maps` instead of the heavier DensePose/SCHP path used by the original project.

### Unpaired preview

The unpaired preview is the better "demo-like" check because the cloth image is different from the original outfit. The final grid shows that the checkpoint generalizes reasonably well on easier cases:

- simple knit tops
- standard pants silhouettes
- dresses with clear structure and moderate texture complexity

Failure cases are also visible, which is expected for this reconstruction:

- sleeves or skirt lengths can be under-defined when the new garment shape differs a lot from the person image
- fine texture sometimes washes out on pale garments
- some category-shape mismatches still look pasted rather than fully re-draped

Overall, the repo is now in a good state for qualitative CatVTON-style validation: the pipeline works end to end, paired results are stable, and unpaired results are useful for manual inspection.

## What is committed to GitHub

These files are intended to make another machine able to run paired and unpaired preview checks immediately after cloning, as long as the DressCode dataset is available locally:

- code changes in `train.py`, `preview_infer.py`, and `catvton_runtime.py`
- final attention checkpoint
- final paired and unpaired preview images
- final `latest.json` metadata pointer for the exact-resume run

## What stays local only

The exact-resume directories under `outputs/full-resume/dresscode-16k-512/training_state/step-*` stay local only.

Reason:

- `accelerate` full training states include multi-hundred-megabyte optimizer files and multi-gigabyte model snapshots
- that is too large for a clean GitHub handoff in this repository

As a result:

- GitHub contains the reusable attention checkpoint for inference and warm-start fine-tuning
- the local machine still holds the exact-resume state if training ever needs to continue here

## Another machine handoff (MPS)

### 1. Clone the repo and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install accelerate diffusers transformers safetensors
```

### 2. Provide the DressCode dataset outside git

Copy the local dataset so that this path exists:

```text
data/DressCode
```

### 3. Regenerate agnostic masks if needed

`agnostic_masks` are not committed. If the copied dataset does not already contain them, run:

```bash
python3 prepare_masks.py \
  --data_root_path data/DressCode \
  --categories upper_body,lower_body,dresses \
  --num_workers 4
```

### 4. Run paired preview on MPS

```bash
python3 preview_infer.py \
  --data_root_path data/DressCode \
  --base_model_path booksforcharlie/stable-diffusion-inpainting \
  --vae_model_path stabilityai/sd-vae-ft-mse \
  --resume_attn_ckpt outputs/full-resume \
  --resume_attn_version dresscode-16k-512 \
  --device mps \
  --mixed_precision no \
  --validation_batch_size 1 \
  --max_val_pairs_per_category 4 \
  --num_inference_steps 30 \
  --guidance_scale 2.5 \
  --split paired \
  --output_path outputs/full-resume/validation/mps-paired-preview.png
```

### 5. Run unpaired preview on MPS

```bash
python3 preview_infer.py \
  --data_root_path data/DressCode \
  --base_model_path booksforcharlie/stable-diffusion-inpainting \
  --vae_model_path stabilityai/sd-vae-ft-mse \
  --resume_attn_ckpt outputs/full-resume \
  --resume_attn_version dresscode-16k-512 \
  --device mps \
  --mixed_precision no \
  --validation_batch_size 1 \
  --max_val_pairs_per_category 4 \
  --num_inference_steps 30 \
  --guidance_scale 2.5 \
  --split unpaired \
  --output_path outputs/full-resume/validation/mps-unpaired-preview.png
```

### 6. Optional: warm-start more training from the committed checkpoint

This is not an exact-resume handoff, because the huge `training_state/step-*` directories are local only. It is still valid to continue with attention-weight warm-starting:

```bash
python3 train.py \
  --data_root_path data/DressCode \
  --base_model_path booksforcharlie/stable-diffusion-inpainting \
  --vae_model_path stabilityai/sd-vae-ft-mse \
  --output_dir outputs/mps-continued \
  --device mps \
  --mixed_precision no \
  --resume_attn_ckpt outputs/full-resume \
  --resume_attn_version dresscode-16k-512 \
  --train_batch_size 1 \
  --validation_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_workers 0 \
  --num_train_steps 200
```

## Final note

For this repo, the GitHub handoff is now optimized for:

- inference preview on another machine
- paired and unpaired qualitative checks
- warm-start continuation from the trained attention checkpoint

Exact optimizer-state resume remains available only on the original RTX machine unless those large `training_state` folders are copied separately outside git.
