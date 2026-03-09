# src/models

CatVTON-inspired 모델 구성 모듈.

- `input_adapter.py`: 13채널 latent concat 입력을 UNet 입력으로 변환
- `unet_patch.py`: self-attention trainable patch 적용
- `pipeline_train.py`: 학습 파이프라인
- `pipeline_infer.py`: 발표용 추론 파이프라인
