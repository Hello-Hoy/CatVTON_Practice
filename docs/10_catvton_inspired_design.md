# 10. CatVTON-inspired Design

## 문서 목적

이 문서는 공식 CatVTON의 학습 코드를 그대로 복원하는 문서가 아니다.  
공식 CatVTON의 핵심 철학을 참고해 **학습 가능한 연구용 재구현 엔진**을 설계하는 문서다.

---

## 핵심 정의

> CatVTON-inspired VTON 엔진은  
> Stable Diffusion inpainting 계열 백본을 기반으로,  
> agnostic person / garment / mask 조건을 latent에서 결합하고,  
> self-attention 중심의 parameter-efficient training으로 의류 try-on 품질을 학습하는 엔진이다.

---

## 설계 철학

### 가져갈 것
- text conditioning 의존 최소화
- concatenation 기반 conditioning
- self-attention 중심 fine-tuning
- inference 입력 단순화

### 바꿀 것
- 학습 루프는 Diffusers 기반으로 직접 재구성
- 데이터셋의 label_maps / keypoints / dense 정보를 전처리에 적극 활용
- 발표 데모 기준으로 inference wrapper를 병렬 설계

---

## v1 입력 정의

학습 샘플은 아래 4개를 기준으로 구성한다.

- `person_img`
- `cloth_img`
- `agnostic_mask`
- `target_img`

추가 파생 입력:
- `agnostic_person_img`

---

## agnostic person 생성 원칙

1. `label_maps`에서 category 의류 영역 추출
2. `keypoints`와 `dense`로 팔/몸통 경계 보정
3. morphology dilation으로 mask 확장
4. head / hair / legs / shoes 보존

---

## 모델 구조 제안

### Base
- Stable Diffusion v1.5 inpainting compatible backbone
- VAE: freeze
- Text encoder: freeze 또는 null prompt
- UNet: 대부분 freeze

### Latent concat
아래를 latent 공간에서 concat 한다.

- `z_noisy` : 4ch
- `z_agnostic_person` : 4ch
- `z_cloth` : 4ch
- `m` : 1ch

총 13채널 입력을 만든다.

### Input adapter
기존 inpainting UNet 입력 형식에 맞추기 위해 trainable input adapter를 둔다.

예시:
- Conv(13 -> 32)
- SiLU
- Conv(32 -> 9)

### Trainable parameters
v1에서는 아래만 학습 가능 대상으로 둔다.

- `input_adapter`
- UNet self-attention (`attn1`) projection
- 필요 시 해당 블록 normalization

나머지는 freeze 한다.

---

## 손실 함수

### v1 필수
- diffusion denoising loss

### v1.1 선택
- garment-region L1
- garment-region LPIPS
- identity perceptual term

초기 구현은 diffusion loss만으로 시작한다.

---

## 학습 범위

### Stage 0
데이터 감사만 수행

### Stage 1
`overfit32`

### Stage 2
pilot subset

### Stage 3
full `upper_body`

---

## v1 해상도/카테고리

- category: `upper_body`
- resolution: `512x384`
- batch size: 1
- grad accumulation: 4~8
- precision: fp16 또는 bf16
- gradient checkpointing: on

---

## v1 성공 기준

1. 32샘플 오버핏이 가능해야 한다.
2. paired validation board에서 garment placement가 명확히 개선되어야 한다.
3. catastrophic artifact 비율이 통제되어야 한다.
4. 학습 체크포인트가 inference wrapper와 연결되어야 한다.
