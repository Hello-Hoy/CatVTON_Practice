# CatVTON_Practice

CatVTON-inspired VTON 엔진을 연구용으로 재구현하고, 학습된 체크포인트를 발표용 쇼핑몰 플러그인 데모에 연결하기 위한 저장소입니다.

## 목표

- DressCode 계열 데이터셋으로 직접 학습 가능한 VTON 엔진 구현
- Gate-based 학습/검증 루프 구축
- 학습된 체크포인트를 사용한 발표용 데모 제작
- 향후 Codex가 이 저장소 구조와 문서를 참고해 구현을 이어갈 수 있도록 기준 문서화

## 저장소 원칙

- 이 저장소는 **코드 + 설정 + 문서 + 리포트** 중심으로 관리합니다.
- 대용량 데이터셋과 대용량 체크포인트는 Git에 직접 커밋하지 않습니다.
- 데이터셋은 로컬 경로에서 참조하고, 체크포인트는 별도 저장소 또는 Git LFS/릴리스 아티팩트로 관리합니다.
- 문서는 `docs/` 아래에서 장기적으로 누적 관리합니다.

## 권장 폴더 구조

```text
CatVTON_Practice/
├─ README.md
├─ .gitignore
├─ docs/
│  ├─ 00_project_architecture.md
│  ├─ 01_repository_conventions.md
│  ├─ 10_catvton_inspired_design.md
│  ├─ 11_training_validation_loop.md
│  └─ 12_dataset_contract.md
├─ configs/
│  ├─ README.md
│  ├─ base_upper_body.yaml
│  ├─ overfit32.yaml
│  ├─ pilot_subset.yaml
│  └─ demo_infer.yaml
├─ scripts/
│  ├─ README.md
│  ├─ audit_data.py
│  ├─ build_agnostic_masks.py
│  ├─ make_fixed_splits.py
│  ├─ train.py
│  ├─ validate.py
│  ├─ benchmark.py
│  └─ export_demo_checkpoint.py
├─ src/
│  ├─ data/
│  │  ├─ README.md
│  │  ├─ dresscode_dataset.py
│  │  ├─ mask_builder.py
│  │  └─ transforms.py
│  ├─ models/
│  │  ├─ README.md
│  │  ├─ input_adapter.py
│  │  ├─ unet_patch.py
│  │  ├─ pipeline_train.py
│  │  └─ pipeline_infer.py
│  ├─ losses/
│  │  ├─ README.md
│  │  ├─ diffusion_loss.py
│  │  └─ optional_region_losses.py
│  ├─ metrics/
│  │  ├─ README.md
│  │  ├─ paired_metrics.py
│  │  ├─ unpaired_metrics.py
│  │  └─ board_builder.py
│  └─ utils/
│     ├─ README.md
│     ├─ seed.py
│     ├─ logging.py
│     └─ checkpointing.py
├─ demo/
│  ├─ README.md
│  ├─ app.py
│  ├─ preprocess_user.py
│  └─ storefront_mock/
├─ data/
│  ├─ README.md
│  └─ local/
├─ reports/
├─ checkpoints/
├─ assets/
└─ tests/
```

## 문서 우선순위

1. `docs/00_project_architecture.md`  
   전체 프로젝트 구조와 실행 흐름
2. `docs/10_catvton_inspired_design.md`  
   CatVTON-inspired 엔진 설계 기준
3. `docs/11_training_validation_loop.md`  
   Codex가 따라야 할 검증/분기 루프
4. `docs/12_dataset_contract.md`  
   데이터셋 구조와 로컬 데이터 계약

## 첫 구현 순서

1. `scripts/audit_data.py` 작성
2. `scripts/build_agnostic_masks.py` 작성
3. `src/data/dresscode_dataset.py` 작성
4. `src/models/input_adapter.py` + `src/models/pipeline_train.py` 작성
5. `scripts/train.py` 에서 `overfit32` 먼저 검증
6. `scripts/validate.py` 와 `src/metrics/board_builder.py` 작성
7. 발표 데모용 `demo/app.py` 연결

## 현재 범위

- 1차 학습 카테고리: `upper_body`
- 1차 해상도: `512x384`
- 1차 목표: 발표 가능한 품질의 학습형 VTON 프로토타입
