# 12. Dataset Contract

## 목적

이 문서는 로컬 DressCode 계열 데이터셋을 코드에서 어떻게 기대할지 정의한다.

---

## 실제 작업 기준

로컬 작업 기준으로는 카테고리별 폴더 아래에 아래 하위 폴더를 기대한다.

- `agnostic_masks`
- `dense`
- `images`
- `keypoints`
- `label_maps`
- `skeletons`

또한 카테고리별 pair txt 파일이 존재해야 한다.

---

## readme 기준 참고 정보

DressCode readme 기준으로 메인 폴더는 category별 폴더와 train/test annotation txt 파일을 포함하고, 각 category는 `images`, `keypoints`, `skeletons`, `label_maps`, `dense`를 기본 서브폴더로 가진다. 또한 category별 train/test 파일로 해당 카테고리만 따로 학습·평가할 수 있고, keypoints는 `512x384` 기준으로 계산되어 있다. fileciteturn4file0L7-L22 fileciteturn4file0L29-L30

readme의 추가 정보에는 파일명 규칙이 명시되어 있다. 모델 이미지는 `_0.jpg`, 의류 이미지는 `_1.jpg`, keypoints는 `_2.json`, skeleton은 `_3.jpg`, label map은 `_4.png`, dense uv/label은 `_5` 계열 파일명 규칙을 따른다. fileciteturn4file1L55-L63

---

## 코드가 기대하는 예시 구조

```text
DressCode/
├─ upper_body/
│  ├─ agnostic_masks/
│  ├─ dense/
│  ├─ images/
│  ├─ keypoints/
│  ├─ label_maps/
│  ├─ skeletons/
│  ├─ train_pairs.txt
│  ├─ test_pairs_paired.txt
│  └─ test_pairs_unpaired.txt
├─ lower_body/
├─ dresses/
└─ ...
```

---

## 코드 계약

### 필수 계약
- pair file의 모든 경로가 실제 파일과 매칭되어야 한다.
- `images` 안에는 model / garment 이미지가 파일명 규칙에 맞게 존재해야 한다.
- `keypoints`, `label_maps`, `dense` 등 보조 파일이 샘플별로 대응되어야 한다.
- `agnostic_masks`가 이미 있더라도, 필요하면 `scripts/build_agnostic_masks.py`로 재생성 가능해야 한다.

### 1차 학습 계약
- 우선 `upper_body` 카테고리만 사용
- 해상도는 `512x384` 기준으로 통일
- 고정 검증셋용 paired/unpaired split을 따로 생성
