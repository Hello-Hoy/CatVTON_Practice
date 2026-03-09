# 11. Training & Validation Loop

## 문서 목적

이 문서는 Codex가 구현 이후 **스스로 검증하고 다음 액션을 선택할 수 있도록** gate-based 루프를 정의한다.

---

## 원칙

모든 실험은 아래 순서로만 진행한다.

1. 구현
2. 단위 검증
3. 시각 검증
4. 소규모 학습
5. 실패 원인 분류
6. 수정
7. 재실행

---

## Gate A — Data Integrity

### 목표
학습 시작 전 데이터 계약을 만족하는지 확인

### 필수 검사
- pair file 파싱
- missing file ratio
- corrupted image ratio
- agnostic mask area ratio
- random visualization 32샘플

### 통과 기준
- missing file ratio < 0.5%
- invalid image ratio = 0
- agnostic mask area가 지나치게 작거나 크지 않음
- 랜덤 샘플에서 심각한 misalignment 3건 이하

### 실패 시 조치
- pair mapping 수정
- naming rule 수정
- mask generation 디버깅

---

## Gate B — Overfit32

### 목표
학습 파이프라인 자체가 올바른지 검증

### 통과 기준
- train loss 30% 이상 감소
- 32개 중 절반 이상에서 garment placement 개선
- NaN / exploding loss 없음

### 실패 유형과 조치

#### 1. 학습이 전혀 안 됨
- latent concat 순서 점검
- input adapter shape 점검
- null prompt 처리 점검
- freeze/trainable 대상 점검

#### 2. 배경만 맞고 옷이 안 바뀜
- garment conditioning 경로 점검
- mask dilate 증가
- cloth latent scaling 점검

#### 3. 팔/목 artifact 심함
- agnostic mask 재생성
- keypoints / dense 보정 강화

---

## Gate C — Pilot Subset

### 목표
작은 subset에서 일반화 조짐 확인

### 통과 기준
- paired fixed set 지표 개선
- visual board artifact 감소
- inference 시간과 VRAM 사용이 현실적

### 실패 시 조치
- OOM: batch 1 유지, grad accumulation 증가, EMA off
- 품질 낮음: LR 감소, self-attn 외 일부 normalization 해제, optional garment-region loss 추가
- 느리지만 품질 좋음: 우선 유지

---

## Gate D — Full Run

### 목표
발표용 best checkpoint 선정

### 통과 기준
- paired / unpaired 고정 보드 안정
- garment fidelity acceptable
- catastrophic artifact rate 낮음
- demo wrapper 연결 성공

---

## 고정 검증셋

절대 바꾸지 말아야 할 셋:

- `overfit32`
- `val_paired_20`
- `val_unpaired_20`

---

## 리포트 산출물

각 run은 아래 파일을 남긴다.

- `reports/run_xxx/summary.md`
- `reports/run_xxx/metrics.json`
- `reports/run_xxx/board_paired.png`
- `reports/run_xxx/board_unpaired.png`
- `reports/run_xxx/decision_log.md`

### `decision_log.md` 필수 항목
- 이번 run에서 무엇을 바꿨는가
- 왜 바꿨는가
- 이전 run 대비 어떤 변화가 있었는가
- 다음 run의 목표는 무엇인가

---

## 발표용 체크포인트 선정 기준

- paired 평균 점수 3.5 이상
- unpaired 평균 점수 3.0 이상
- catastrophic failure 비율 20% 이하
- demo inference가 안정적으로 1회 이상 연속 수행됨
