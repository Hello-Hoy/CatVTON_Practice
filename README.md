# CatVTON Practice

DressCode dataset를 이용해 CatVTON 방식의 self-attention-only 학습을 재구성하는 프로젝트입니다.

이 저장소는 두 단계를 기준으로 설계되어 있습니다.

- 로컬 MacBook(MPS)에서 `공개 CatVTON DressCode attention checkpoint`를 warm-start로 불러와 smoke fine-tune 및 샘플 검증
- RTX 4080 Super CUDA 환경에서 같은 코드로 DressCode 전체 학습

## 포함 파일

- `train.py`
  - device-aware 학습 루프
  - CatVTON 공개 checkpoint warm-start 지원
  - subset 학습과 full 학습을 동일 CLI에서 제어
  - step 0 baseline validation 지원
- `prepare_masks.py`
  - DressCode `label_maps` 기반 `agnostic_masks` 선생성
- `preview_infer.py`
  - 어떤 attention checkpoint든 고정된 paired validation 샘플 세트로 미리보기 생성
- `.codex/skills/catvton-train-reconstruction`
  - 이 프로젝트에서 train 코드를 확장하거나 검토할 때 사용하는 로컬 Codex skill

## 데이터셋

기본 경로는 `data/DressCode` 입니다.

현재 로컬 데이터는 다음 구조를 사용합니다.

- `<category>/images`
- `<category>/label_maps`
- `<category>/train_pairs.txt`
- `<category>/test_pairs_paired.txt`
- `<category>/test_pairs_unpaired.txt`

DressCode 원본에는 `agnostic_masks`가 없으므로, 이 프로젝트는 `label_maps`로부터 category-aware mask를 만들어 `<category>/agnostic_masks`에 캐시합니다.

## 의존성

```bash
pip install -r requirements.txt
```

## 로컬 smoke 절차

### 1. 마스크 생성

```bash
python3 prepare_masks.py \
  --data_root_path data/DressCode \
  --categories upper_body,lower_body,dresses \
  --num_workers 4
```

### 2. warm-start baseline preview

```bash
python3 preview_infer.py \
  --data_root_path data/DressCode \
  --resume_attn_ckpt zhengchong/CatVTON \
  --resume_attn_version dresscode-16k-512 \
  --device mps \
  --mixed_precision no \
  --max_val_pairs_per_category 4 \
  --output_path outputs/smoke/validation/step-000000-preview.png
```

### 3. 100-step smoke fine-tune

```bash
python3 train.py \
  --data_root_path data/DressCode \
  --output_dir outputs/smoke \
  --device mps \
  --mixed_precision no \
  --resume_attn_ckpt zhengchong/CatVTON \
  --resume_attn_version dresscode-16k-512 \
  --train_batch_size 1 \
  --validation_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_workers 0 \
  --num_train_steps 100 \
  --checkpointing_steps 50 \
  --validation_steps 50 \
  --validation_num_inference_steps 10 \
  --run_validation_at_start \
  --max_train_pairs_per_category 64 \
  --max_val_pairs_per_category 4
```

산출물:

- `outputs/smoke/validation/step-000000.png`
- `outputs/smoke/validation/step-000050.png`
- `outputs/smoke/validation/step-000100.png`
- `outputs/smoke/dresscode-16k-512/attention`

### 4. smoke checkpoint 재검증

```bash
python3 preview_infer.py \
  --data_root_path data/DressCode \
  --resume_attn_ckpt outputs/smoke \
  --resume_attn_version dresscode-16k-512 \
  --device mps \
  --mixed_precision no \
  --max_val_pairs_per_category 4 \
  --output_path outputs/smoke/validation/post-train-preview.png
```

## RTX 4080 Super full 학습 절차

먼저 이 프로젝트를 데이터 제외 상태로 다른 머신에 옮깁니다.

- `.gitignore`가 `data/`, `outputs/`, `agnostic_masks/` 등을 제외하도록 설정되어 있습니다.
- 다른 머신에서는 동일한 `data/DressCode` 경로 또는 대응 경로를 준비하면 됩니다.

추천 full run:

```bash
python3 prepare_masks.py \
  --data_root_path data/DressCode \
  --categories upper_body,lower_body,dresses \
  --num_workers 16
```

```bash
python3 train.py \
  --data_root_path data/DressCode \
  --output_dir outputs/full \
  --device cuda \
  --mixed_precision bf16 \
  --resume_attn_ckpt zhengchong/CatVTON \
  --resume_attn_version dresscode-16k-512 \
  --train_batch_size 2 \
  --validation_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_workers 8 \
  --num_train_steps 16000 \
  --checkpointing_steps 500 \
  --validation_steps 500
```

먼저 20-step dry run, 이후 500-step 확인, 그 다음 full 16k로 확장하는 것을 권장합니다.

## 체크포인트 형식

attention checkpoint는 CatVTON 공개 형식과 맞춰 다음 경로에 저장됩니다.

```text
<output_dir>/<dataset_tag>/attention
```

`resume_attn_ckpt`는 다음 둘 다 지원합니다.

- Hugging Face repo id 예: `zhengchong/CatVTON`
- 로컬 경로 예: `outputs/smoke`

## GitHub 업로드 준비

이 폴더는 데이터 제외 상태로 커밋하도록 설계되어 있습니다.

기본 절차:

```bash
git init
git add .
git commit -m "Initialize CatVTON DressCode training project"
```

## 주의

- 이 구현은 CatVTON 논문과 공개 checkpoint 구조를 기준으로 한 재구현입니다.
- 로컬 MPS smoke는 품질 검증보다 파이프라인 검증과 warm-start 비교에 초점을 둡니다.
- 최종 qualitative quality 확보는 CUDA 환경에서 수행하는 것을 전제로 합니다.
