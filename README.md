# Document Classification (CV)

## 1. Overview

본 프로젝트는 17개 클래스의 문서 이미지를 분류하는 Computer Vision 문제를 해결하는 것을 목표로 한다.  
EfficientNet 기반 모델과 ensemble 전략을 적용하여 Leaderboard 기준 약 0.86 수준의 성능을 달성했다.

- Task: Document Image Classification (17 classes)
- Metric: Macro F1
- Train: 1570 images
- Test: 3140 images

---

## 2. Directory

```bash
cv_project/
├── train.py
├── inference.py
├── requirements.txt
├── README.md
├── data/
│   ├── train/
│   ├── test/
│   ├── sample_submission.csv
│   ├── hard_test/
│   └── sample_imgs/
├── checkpoints/
├── logs/
├── outputs/
│   ├── experiments/
│   ├── submissions/
│   └── final/

```
---

## 3. Data Description

### 3.1 Dataset Overview

- 총 17개 클래스 문서 이미지 분류 문제
- Train: 1570장
- Test: 3140장
- 다양한 도메인의 문서 포함 (금융, 의료, 일반 문서 등)
- 텍스트 중심 / 표 / 혼합 구조 이미지 존재

---

### 3.2 EDA

데이터 분석을 통해 다음 특징을 확인했다.

- 문서 방향이 일정하지 않음 (Rotate 필요)
- 일부 이미지에 강한 노이즈 존재
- 문서 외 객체(차량, 계기판 등) 혼입 데이터 존재
- 다양한 해상도 및 비율
- 클래스별 데이터 분포 불균형
- 일부 클래스 간 시각적 유사도가 높아 classification difficulty 존재

따라서 모델 학습 시 다음과 같은 전략이 필요하다고 판단했다.

- Rotation / Flip 기반 augmentation 필수
- 노이즈 대응 augmentation 필요
- backbone의 generalization 성능 중요

---

## 4. Solution

### 4.1 Backbone

- EfficientNet-B4

---

### 4.2 Training

- Stratified K-Fold (5-fold)
- Multi-seed (42, 777, 2024)
- AMP (mixed precision)
- Class imbalance 고려 (weighted loss 적용)
- 
---

### 4.3 Augmentation

- Resize / Normalize
- Flip / Rotation
- ColorJitter / Blur

---

### 4.4 Ensemble

- Fold ensemble
- Seed ensemble
- Logit averaging

---

## 5. Techniques

본 프로젝트에서 실제 적용한 핵심 기법은 다음과 같다.

- Stratified K-Fold
- Multi-seed training
- Mixed Precision Training (AMP)
- Test Time Augmentation (Flip)
- Logit averaging ensemble

---

## 6. Results

| Model | Public LB |
|------|----------|
| ResNet18 KFold | val 0.83 / LB 0.05 (설정 오류)|
| EfficientNet-B0 | 0.46 |
| EfficientNet-B0 + seed | 0.65 |
| ConvNeXt | 약 0.65 |
| EfficientNet-B4 | 0.861 |
| EfficientNet-B4 + TTA + ensemble | 0.8633 |

최종적으로 EfficientNet-B4 기반 fold ensemble과 flip TTA 조합이 가장 높은 성능을 기록했다.

---

## 7. How to Run

```bash
pip install -r requirements.txt

python train.py
python inference.py

# 결과 CSV 생성
# outputs/submissions/ 에 저장됨
