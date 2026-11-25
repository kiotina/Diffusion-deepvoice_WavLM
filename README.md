# Real-only Diffusion 딥보이스 탐지

진짜 음성(bona fide)만으로 디퓨전 모델(DDPM)을 학습하고, **노이즈를 얼마나 못 맞히는지(노이즈 예측 오차)**를 이상 점수로 써서 가짜 음성(spoof)을 탐지함.
학습, 표준화, 점수 보정, 임계값까지 모든 단계에서 **가짜 음성을 사용하지 않음** (real-only). 가짜 음성은 마지막 평가에서 처음 봄.

> 진짜만 보고 배운 디퓨전은 진짜 같은 입력의 노이즈는 잘 맞히고, 낯선 입력의 노이즈는 더 많이 틀린다.
> 그 "틀린 정도"가 처음 보는 진짜 음성들보다 확실히 크면 fake로 판정한다.

## 결과

시연 모델 (`rdm_xlsr_l22_en_libri`), 임계값 = dev real 점수의 90백분위

| 평가 데이터 | 조건 | EER | AUROC | 오탐률 | 탐지율 |
|---|---|---|---|---|---|
| **ASVspoof 2019 LA eval** (8,000개 표본) | 깨끗한 녹음, 공격 A07–A19 | **13.01%** | **0.939** | 2.1% | 70.4% |
| 같은 표본 + 방 환경(울림·잡음)을 입힘 | 방에서 녹음한 소리 흉내 | 23.92% | 0.839 | 0.3% | 7.9% |
| **ASVspoof 5 eval** (코덱 없는 2,496개) | 최신 TTS·음성 변환, 공격 A17–A32 | **17.24%** | **0.902** | 27.4% | 86.6% |

- 학습 seed 3개 평균 2019 EER **12.74%** (13.01 / 12.33 / 12.88%)
- 논문용 모델(2019 train real만, 증강 없음)은 2019 EER **10.87%** (seed 3개 평균). 대신 방 환경·ASVspoof 5에는 약함
- EER·AUROC: real과 fake를 얼마나 잘 가르는지 (임계값과 무관)
- 오탐률·탐지율: dev real 90백분위 임계값으로 실제 판정했을 때. real의 약 10%까지는 의심해도 감수하는 대신 fake를 덜 놓치도록 정함
- 평가의 공격과 화자는 모두 학습에 없음

### 같은 벡터의 다른 이상 탐지 방법과 비교 (2019 / 방 환경 / ASVspoof 5 EER)

| 방법 | 2019 | 방 환경 | ASVspoof 5 |
|---|---|---|---|
| 평균까지 거리 | 43.52% | 55.66% | 45.41% |
| 마할라노비스 | 14.14% | 26.70% | 16.16% |
| k-NN (k=5) | 16.55% | 30.53% | **15.79%** |
| GMM (2성분) | 13.36% | 27.41% | 16.47% |
| One-class SVM | 27.44% | 49.55% | 20.65% |
| **디퓨전 (RDM, seed 3개 평균)** | **12.74%** | **23.23%** | 17.35% |

디퓨전이 2019와 방 환경에서 가장 좋지만, 2019는 GMM과 같은 수준(차이 0.6%p). 파일 평균 벡터의 real 분포가 가우시안 몇 개로 거의 설명되기 때문으로 해석함.

## 모델 구조

```
음성 (16kHz)
 → 앞뒤 무음 자르기 (가장 큰 소리보다 40dB 작은 구간) + 최대 진폭 1.0
 → XLS-R 300M 22층 (가중치 고정, 20ms마다 1024개 숫자)
 → 시간 평균 → 파일 벡터 1024개 → train real 기준 표준화 = x₀
 → 디퓨전 (MLP, real만으로 학습): x₀에 노이즈 ε를 섞고 섞인 노이즈를 맞힘
 → t = 50, 100, 200 × 고정 노이즈 16개의 예측 오차
 → dev real 기준 z 점수의 평균 = 파일 점수
 → dev real 점수의 90백분위보다 크면 fake
```

| 단계 | 파일 | 하는 일 |
|---|---|---|
| 전처리 | `preprocess.py` | 음성 읽기, 무음 자르기·크기 맞춤, 학습 real 증강(방 환경 또는 배속), XLS-R 22층 → 시간 평균 벡터 저장 |
| 학습 | `train.py` | train real 벡터 표준화, DDPM(T=1000, 선형 β, ε 예측, MSE)으로 MLP 학습. dev real 손실이 가장 낮은 모델을 `best.pt`로 저장 |
| 평가 | `evaluate.py` | t별 오차 → dev real로 보정·임계값 → 2019 eval(깨끗함·방 환경), ASVspoof 5 지표 + 시험지 판정 |

- **XLS-R** (`facebook/wav2vec2-xls-r-300m`): 128개 언어 약 43.6만 시간으로 자기지도 학습된 공개 음성 모델. 학습하지 않고 특징 추출에만 사용. 내 컴퓨터 GPU에서 돌아감 (처음 한 번 가중치만 내려받음)
- **MLP 디퓨전**: 입력 Linear → 잔차 블록 3개(LayerNorm → Linear → +t → SiLU → Dropout → Linear, skip) → LayerNorm·SiLU·Linear. 폭은 처음부터 끝까지 1024, 가중치 약 880만 개
- **무음 자르기**: ASVspoof 2019는 real과 fake의 앞뒤 무음 길이가 다름. 무음을 남기면 모델이 합성 흔적 대신 무음 길이로 판단함 (예전 WavLM 실험: 무음 그대로 EER 14% → 자르니 25.6%)

## 폴더 구조

```
PBL2_prototype/
├─ src/
│   ├─ config.py        경로와 모든 설정값 (실험할 때는 이 파일만 수정)
│   ├─ preprocess.py    음성 → 파일 벡터 (무음 자르기, 증강, XLS-R)
│   ├─ train.py         MLP 디퓨전 학습
│   ├─ evaluate.py      점수, 임계값, EER·AUROC, 공격별 결과, 시험지 판정
│   ├─ explain.py       음성 하나를 분석해 판정과 근거(2초 구간별 점수, t별 점수)를 만듦
│   └─ server.py        웹 서버 (음성을 받아 explain.py로 분석)
├─ web/
│   └─ index.html       웹 화면
├─ outputs/             실행 결과 (Git 제외)
├─ requirements.txt
└─ README.md
```

## 설치 (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install soundfile==0.12.1
```

- `soundfile`은 **0.12.1로 고정**. 0.13 이상은 ASVspoof 2021 flac 파일의 일부를 읽지 못함
- XLS-R은 처음 실행할 때 Hugging Face에서 약 1.2GB를 자동으로 내려받음. 인터넷 없이 실행하려면 미리 한 번 실행해 둔 뒤 `$env:HF_HUB_OFFLINE = "1"`
- 개발 환경: Python 3.11, PyTorch (CUDA 12.8), RTX 3070 Ti 8GB

## 데이터

경로는 `config.py`의 `DATASET_ROOT`와 `DATASETS`에서 수정.

| 데이터 | 사용하는 것 | 용도 |
|---|---|---|
| [ASVspoof 2019 LA](https://doi.org/10.7488/ds/2555) train | real 2,580개 (+ 증강본) | 학습 |
| LibriSpeech dev-clean | real 2,703개 (+ 증강본) | 학습 (오디오북 녹음 조건의 real) |
| ASVspoof 2019 LA dev | real 2,548개 | 검증, 점수 보정, 임계값 |
| LibriSpeech test-clean | real 2,620개 | 검증, 점수 보정, 임계값 (학습과 다른 화자) |
| ASVspoof 2019 LA eval | real + fake 8,000개 (seed 42 무작위 표본) | 최종 평가 |
| ASVspoof 5 eval | 내려받은 12,159개 중 코덱 없는 2,496개 (real 646) | 최신 공격 평가 |

train·dev의 fake는 사용하지 않음. ASVspoof 5의 real도 오디오북(LibriVox) 계열이라, LibriSpeech를 학습에 넣은 효과가 섞여 있음.

## 실행

프로젝트 맨 위 폴더에서, venv를 켠 상태로 실행.

```powershell
python src/preprocess.py   # 음성 → 파일 벡터 (처음 한 번, 약 1시간)
python src/train.py        # 학습 (약 5분)
python src/evaluate.py     # 평가 + 시험지 (약 30초)
```

웹에서 직접 음성을 분석하려면 (학습한 모델 필요)

```powershell
python src/explain.py      # 처음 한 번 (2초 구간 점수의 기준값 만들기, 모델이 바뀔 때마다)
python src/server.py       # 브라우저에서 http://127.0.0.1:8000
```

### 모델 고르기 (`src/config.py`)

| 설정 | 시연 모델 (지금 값) | 논문 모델 |
|---|---|---|
| `RUN_NAME` | `"rdm_xlsr_l22_en_libri"` | `"rdm_xlsr_l22"` |
| `TRAIN_DATA` | `["2019_train", "libri_train"]` | `["2019_train"]` |
| `DEV_DATA` | `["2019_dev", "libri_dev"]` | `["2019_dev"]` |
| `AUGMENT` | `True` (방 환경 또는 배속 증강본 1개) | `False` |

### 결과 파일 (`outputs/<RUN_NAME>/`)

| 파일 | 내용 |
|---|---|
| `checkpoints/best.pt` | 학습한 모델 + 표준화 통계 |
| `checkpoints/history.csv`, `loss_curve.png` | 500걸음마다 train/dev 손실 |
| `evaluation_2019_eval/calibration.json` | dev real 기준 t별 평균·표준편차와 임계값 |
| `evaluation_2019_eval/metrics_<데이터>_<clean\|room>.json` | EER, AUROC, 오탐률·탐지율, 공격별 결과 |
| `evaluation_2019_eval/predictions_<데이터>_<clean\|room>.csv` | 파일별 점수와 판정 |
| `evaluation_2019_eval/score_hist_<데이터>_<clean\|room>.png` | real·fake 점수 분포와 임계값 |
| `evaluation_2019_eval/test_samples.csv` | 시험지(`test_samples/`) 파일별 점수와 판정 |

## 실험 과정 요약

2019 eval 기준. 자세한 숫자는 실험 기록에 있음.

| 단계 | 입력 → 판정 | 2019 EER | 배운 것 |
|---|---|---|---|
| Mel + 2D U-Net | 스펙트로그램 디퓨전 | 28.77% | 무음 길이 차이로 맞힌 몫이 섞여 있었음 |
| WavLM 3층 + 1D U-Net | 무음 포함 | 14.10% | 무음을 자르니 25.56%로 떨어짐 → 무음 지름길이었음 |
| 스펙트로그램·위상 디퓨전 | 무음 자름 | 33~45% | 오차가 "진짜다움"보다 "소리의 복잡도"를 따라감 |
| XLS-R 프레임 디퓨전 | 프레임마다 디퓨전 | 약 20% | 프레임별 오차가 소리 복잡도에 흔들림 |
| **XLS-R 파일 벡터 + MLP 디퓨전 (RDM)** | 파일 평균 벡터 | **11.49%** (2019만) | 파일 평균으로 흔들림이 사라짐 |
| + LibriSpeech real, 방 환경·배속 증강 | 시연 모델 | 13.01% | 방 환경·최신 공격에서 크게 좋아지는 대신 2019는 조금 나빠짐 |

## 한계

- **잡지 못하는 공격**: 2019 A10(AUROC 0.58), ASVspoof 5 A28(0.39, fake 점수가 오히려 낮음)·A29·A17. 이 특징 공간에서 순서로도 잘 갈리지 않음
- **방 환경**: EER은 23.92%로 버티지만, 같은 임계값에서 탐지율이 7.9%로 떨어짐. 방 울림이 합성 흔적을 덮음
- **real 출처마다 점수 위치가 다름**: dev 점수 중앙값이 2019 −0.45, LibriSpeech +0.14. 임계값 하나로 모든 출처를 맞추기 어려움 (LibriSpeech dev 오탐 15.4%)
- **짧은 음성**: 앞뒤 무음을 뺀 말소리가 1.5초 미만이면 real 오탐이 약 17%. 웹은 2초 미만이면 경고를 띄움
- **디퓨전의 이점**: 같은 벡터에서 마할라노비스·GMM과 비슷한 수준. 방 환경처럼 real 분포가 복잡해질 때 차이가 남
- **eval을 보고 정한 것** (결과를 보기 전에 정하지 못함): XLS-R 층 후보(19~24층) 비교, 임계값 95 → 90백분위 변경
