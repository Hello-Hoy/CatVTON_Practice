# Repo Notes

## Dataset assumptions

- Root dataset path: `data/DressCode`
- Categories:
  - `upper_body`
  - `lower_body`
  - `dresses`
- Each category contains:
  - `images`
  - `label_maps`
  - `dense`
  - `keypoints`
  - `skeletons`
  - `train_pairs.txt`
  - `test_pairs_paired.txt`
  - `test_pairs_unpaired.txt`

## Local implementation choice

This practice repo generates `agnostic_masks` from `label_maps` and caches them under each category folder.

The mask classes are category-aware and are intentionally simple:

- `upper_body`: upper clothes and arms
- `lower_body`: skirt, pants, belt, legs
- `dresses`: torso, skirt, pants, dress, arms, legs

## Checkpoint layout

Training saves to:

```text
outputs/
  dresscode-16k-512/
    attention/
```

This mirrors the public CatVTON attention-only checkpoint organization closely enough for follow-up inference tooling.
