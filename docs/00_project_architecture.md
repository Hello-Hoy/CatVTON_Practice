# 00. Project Architecture

## 목적

이 문서는 `CatVTON_Practice` 저장소의 **전체 프로젝트 아키텍처**를 정의한다.  
이 저장소의 목표는 다음 하나로 요약된다.

> DressCode 계열 데이터셋으로 학습 가능한 CatVTON-inspired VTON 엔진을 재구현하고,  
> 그 결과 체크포인트를 발표용 쇼핑몰 플러그인 데모로 연결한다.

---

## 프로젝트 레이어

프로젝트는 아래 4개 레이어로 나눈다.

### 1. Documentation Layer
- 프로젝트 정의
- 설계 의도
- 데이터 계약
- 검증 기준
- 실험 의사결정 로그

### 2. Training Layer
- 데이터 감사
- agnostic mask 생성
- dataset loader
- 학습 파이프라인
- checkpoint 저장

### 3. Validation Layer
- paired / unpaired 검증
- 시각 보드 생성
- 정량 지표 계산
- 실패 원인 분류 및 다음 액션 결정

### 4. Demo Layer
- 사용자 사진 업로드
- 상품 의류 선택
- 전처리
- inference 호출
- 결과 이미지 반환

---

## 권장 실행 흐름

```text
Dataset Audit
  -> Agnostic Mask Build / Verify
  -> Fixed Split Creation
  -> Overfit32
  -> Pilot Subset
  -> Full Category Training
  -> Validation Boards + Metrics
  -> Best Checkpoint Selection
  -> Demo Inference Wrapper
  -> Storefront Mock Demo
```

---

## 저장소별 역할

### `docs/`
프로젝트의 기준 문서를 저장한다.  
Codex는 구현 전 항상 `docs/10_catvton_inspired_design.md` 와 `docs/11_training_validation_loop.md` 를 먼저 읽고 시작한다.

### `configs/`
실험 설정을 관리한다.
- `base_upper_body.yaml`: 기본 실험 설정
- `overfit32.yaml`: 초소형 디버깅 설정
- `pilot_subset.yaml`: 중간 검증 설정
- `demo_infer.yaml`: 발표 데모용 추론 설정

### `scripts/`
실행 엔트리포인트를 둔다.
- 데이터 감사
- mask 생성
- split 생성
- train / validate / benchmark

### `src/`
실제 로직을 모듈화한다.
- `data/`: 데이터 로더와 변환
- `models/`: UNet patch, input adapter, train/infer pipeline
- `losses/`: diffusion 및 선택 손실
- `metrics/`: paired/unpaired 평가 및 보드 생성
- `utils/`: logging, checkpointing, seed 등

### `demo/`
발표용 UI와 사용자 전처리를 둔다.

### `reports/`
각 실험의 결과 요약을 저장한다.

### `checkpoints/`
모델 체크포인트 저장 위치다. Git 직접 커밋 대상이 아니다.

---

## 실험 원칙

### 원칙 1. 문서보다 코드가 먼저가 아니다
코드 작성 전, 데이터 계약과 검증 게이트를 먼저 정의한다.

### 원칙 2. Full run 이전에 Overfit32 통과가 필수다
32샘플 오버핏 검증에 실패한 상태에서는 전체 학습을 시작하지 않는다.

### 원칙 3. Demo는 학습 이후가 아니라 병렬 준비다
학습 결과가 나오면 즉시 UI에 연결할 수 있게 inference wrapper와 demo 구조를 초기에 같이 만든다.

### 원칙 4. 결과 선택은 같은 샘플셋 기준으로 한다
고정된 paired / unpaired validation board를 사용한다.

---

## v1 범위

- 데이터셋: DressCode 계열 로컬 데이터
- 카테고리: `upper_body`
- 해상도: `512x384`
- 목표: 발표 가능한 학습형 VTON 프로토타입
- 비목표: 논문 수치 완전 재현, 전 카테고리 동시 학습, 상용 배포 최적화

---

## 발표 성공 기준

1. 직접 학습된 체크포인트가 존재한다.
2. paired / unpaired 고정 보드에서 baseline 대비 개선이 보인다.
3. inference wrapper가 학습 체크포인트를 정상 로드한다.
4. 사용자 사진 업로드 + 의류 선택 + 결과 반환 데모가 동작한다.
