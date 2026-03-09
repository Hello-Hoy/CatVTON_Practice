# 01. Repository Conventions

## 브랜치 규칙

- `main`: 문서 기준선 + 안정 코드
- `feature/data-audit`
- `feature/mask-builder`
- `feature/train-loop`
- `feature/demo-app`

## 파일 명명 규칙

- 설정 파일: `snake_case.yaml`
- 파이썬 모듈: `snake_case.py`
- 리포트 폴더: `reports/run_YYYYMMDD_HHMM_*`
- 체크포인트: `ckpt_stepXXXX.pt` 또는 `checkpoint-stepXXXX`

## 실험 로그 규칙

각 실험은 아래 파일을 남겨야 한다.

- `summary.md`
- `metrics.json`
- `board_paired.png`
- `board_unpaired.png`
- `decision_log.md`

## 금지 사항

- 대용량 데이터셋을 Git에 직접 커밋하지 않는다.
- 검증 보드 없이 “좋아 보인다”는 이유로 best checkpoint를 선택하지 않는다.
- Gate A~D를 생략하고 full training으로 넘어가지 않는다.
