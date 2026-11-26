# Real-only Diffusion 딥보이스 탐지 모델 개발

진짜 음성(bona fide)만으로 디퓨전 모델(DDPM)을 학습하고, **노이즈 예측 오차**를 이상 점수로 사용해 가짜 음성(spoof)을 탐지함.
학습, 정규화, 임계값 설정까지 모든 단계에서 **가짜 음성을 사용하지 않음** (real-only).

## 웹 데모

모델 없이 화면만 볼 수 있음 (GitHub Pages)

- 화면 보기: https://kiotina.github.io/Diffusion-deepvoice_WavLM/web/
- 예시 결과 보기: https://kiotina.github.io/Diffusion-deepvoice_WavLM/web/?demo=1

GitHub Pages에는 분석 서버가 없어서 실제 음성 분석은 동작하지 않음 (화면 위에 "분석 서버에 연결되지 않음"으로 표시됨).
주소 뒤에 `?demo=1`을 붙이면 예시 데이터로 결과 화면(판정, 구간별 점수 등)을 볼 수 있음.
실제 분석은 아래 「실행」의 웹 서버를 내 컴퓨터에서 켜야 가능함.

## 결과

ASVspoof LA eval 전체, 점수 `score_mean`, 임계값 dev real 95%

| 평가 데이터 | 조건 | EER | AUROC | 오탐률 | 탐지율 |
|---|---|---|---|---|---|
| **ASVspoof 2019 LA** (71,237개) | 깨끗한 녹음 | **14.10%** | **0.930** | 4.2% | 61.8% |
| ASVspoof 2021 LA (181,566개) | 전화망·코덱을 거친 음성 | 29.41% | 0.789 | 50.4% | 87.3% |
| 2021 중 코덱 없음, 무음 그대로 | 2019와 같은 조건 | 15.96% | 0.923 | 4.5% | 66.5% |

- 평가의 공격(A07–A19)과 화자는 모두 학습에 없음
- EER·AUROC: 모델이 real과 fake를 얼마나 잘 가르는지
- 오탐률·탐지율: dev real 95% 임계값으로 실제 판정했을 때의 결과
- 2021은 2019로 학습한 모델을 그대로 평가함. 코덱마다 순서는 잘 가르지만(코덱 안 EER 15~22%) 전화 코덱의 real 점수가 높아져 임계값이 어긋남

## 모델 구조

```
음성 (16kHz)
 → WavLM 3층 특징 (20ms마다 768개 숫자)
 → 무음 프레임 제외 (가장 큰 소리보다 40dB 작은 프레임)
 → real-only 디퓨전: 노이즈를 섞고 섞인 노이즈를 맞히도록 학습 (1D U-Net)
 → 노이즈 예측 오차 = 이상 점수 (score_mean)
 → dev real 점수의 95% 지점보다 크면 fake
```

| 단계 | 파일 | 하는 일 |
|---|---|---|
| 전처리 | `preprocess.py` | 음성을 읽어 WavLM 특징과 무음 mask를 만들고, train real 통계로 정규화해 저장 |
| 학습 | `train.py` | train real만으로 DDPM 학습 (T=1000, 선형 β, ε 예측, masked MSE). dev real loss가 가장 낮은 모델을 `best.pt`로 저장 |
| 평가 | `evaluate.py` | 파일을 2초 창·1초 hop으로 자르고, t = 100·300·500에서 고정 노이즈로 오차 측정. dev real 기준으로 표준화해 점수를 내고, 임계값과 EER·AUROC 계산 |

- **WavLM** (`microsoft/wavlm-base-plus`): 진짜 사람 음성으로만 미리 학습된 음성 모델. 학습하지 않고 특징 추출에만 사용
- **1D U-Net**: WavLM의 768차원에는 주파수처럼 위아래 순서가 없음. 768개를 채널로 보고 시간 방향으로만 conv 적용
- **무음 제외**: ASVspoof 2019 LA는 real과 fake의 앞뒤 무음 길이가 다름. 무음을 남기면 모델이 합성 흔적 대신 무음 길이로 판단함

## 폴더 구조

```
PBL2_prototype/
├─ src/
│   ├─ config.py        경로와 모든 설정값 (실험할 때는 이 파일만 수정)
│   ├─ preprocess.py    음성 → WavLM 특징(또는 Mel·스펙트로그램), 무음 mask, 정규화, 저장
│   ├─ train.py         DDPM 학습 (WavLM용 1D U-Net, Mel·스펙트로그램용 2D U-Net)
│   ├─ evaluate.py      이상 점수, 임계값, EER·AUROC, 공격별 결과
│   ├─ explain.py       음성 하나를 분석해 판정과 구간별 설명을 만듦
│   └─ server.py        웹 서버 (음성을 받아 explain.py로 분석)
├─ web/
│   └─ index.html       웹 화면 (?demo=1로 서버 없이 예시 보기)
├─ outputs/             실행 결과 (Git 제외). 실험마다 outputs/<RUN_NAME>/에 저장
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

- `soundfile`은 **0.12.1로 고정**. 0.13 이상은 ASVspoof 2021 flac 파일의 약 40%를 읽지 못함
- WavLM 모델은 처음 실행할 때 Hugging Face에서 자동으로 내려받음
- 개발 환경: Python 3.11, PyTorch (CUDA 12.8), RTX 3070 Ti 8GB

## 데이터

[ASVspoof 2019 LA](https://doi.org/10.7488/ds/2555)와 [ASVspoof 2021 LA](https://doi.org/10.5281/zenodo.4837263) 사용.
경로는 `config.py`의 `DATASET_ROOT`와 `DATASETS`에서 수정.

| 분할 | 사용하는 것 | 용도 |
|---|---|---|
| 2019 train | real 2,580개 | 학습 |
| 2019 dev | real 2,548개 | 검증, 점수 표준화, 임계값 |
| 2019 eval | real + fake 71,237개 | 최종 평가 |
| 2021 eval | real + fake 181,566개 | 녹음 환경이 바뀐 조건에서의 평가 (정답은 `keys/LA/CM/trial_metadata.txt`) |

train·dev의 fake는 사용하지 않음.

## 실행

프로젝트 맨 위 폴더에서, venv를 켠 상태로 실행.

```powershell
python src/preprocess.py   # 약 5분
python src/train.py        # 약 12분 (100 epoch)
python src/evaluate.py     # 2019 eval 전체 약 1시간 10분
```

웹에서 직접 음성을 분석하려면 (학습한 모델 필요)

```powershell
python src/explain.py      # 처음 한 번 (기준값 만들기)
python src/server.py       # 브라우저에서 http://127.0.0.1:8000
```

### 자주 바꾸는 설정 (`src/config.py`)

| 설정 | 최종 값 | 설명 |
|---|---|---|
| `RUN_NAME` | `"wavlm_l3"` | 결과 폴더 이름. 설정을 바꿔 새로 실험할 때는 이름도 변경 |
| `FEATURE_TYPE` | `"wavlm"` | 디퓨전 입력: `"mel"`, `"spec"`, `"wavlm"` |
| `WAVLM_LAYER` | `3` | WavLM에서 꺼낼 층 (0 = CNN 출력, 1~12 = Transformer 층) |
| `MASK_SILENCE` | `True` | 무음 프레임을 학습·점수에서 제외 |
| `AUGMENT_CODEC` | `False` | 코덱 증강 (실험 후 사용하지 않기로 함, 아래 참고) |
| `EVAL_DATA` | `"2019_eval"` | 평가 데이터 (`"2021_eval"`로 바꾸면 2021 평가) |
| `EVAL_LIMIT` | `None` | 숫자를 넣으면 eval에서 무작위로 그 개수만 평가 (빠른 확인용) |

### 결과 파일 (`outputs/<RUN_NAME>/`)

| 파일 | 내용 |
|---|---|
| `checkpoints/best.pt` | 학습한 모델 |
| `checkpoints/history.csv`, `loss_curve.png` | epoch별 loss |
| `evaluation_<EVAL_DATA>/metrics.json` | EER, AUROC, 오탐률·탐지율, 공격별 결과 |
| `evaluation_<EVAL_DATA>/predictions.csv` | 파일별 점수와 판정 |
| `evaluation_<EVAL_DATA>/segments.csv` | 2초 구간별 오차 (중간에 멈춰도 이어서 실행 가능) |
| `evaluation_<EVAL_DATA>/score_hist.png` | real·fake 점수 분포와 임계값 |

## 실험 기록

2019 LA eval 전체 기준

| 단계 | 바꾼 것 | EER | 결과 |
|---|---|---|---|
| baseline | Mel 80칸, 무음 포함 | 28.77% | 무음 길이 차이로 맞힌 몫이 섞여 있음 |
| 무음 제외 | `MASK_SILENCE = True` | 57.57% | 말소리만 보면 fake 오차가 오히려 작아 판단 방향이 뒤집힘 |
| 마할라노비스 점수 | 방향을 가정하지 않는 점수 | 38.99% | |
| EMA, 학습 t 범위 | `USE_EMA`, `TRAIN_T_MAX` | 39.09–39.49% | 효과 없음 (real·fake 오차가 함께 줄어듦) |
| 스펙트로그램 | Mel 대신 log 스펙트로그램 256칸 | 32.97% | |
| WavLM 2층 | WavLM 특징 + 1D U-Net | 18.02% | fake 오차가 다시 real보다 커져 단순한 `score_mean`이 가장 좋음 |
| 코덱 증강 | `AUGMENT_CODEC = True` | 23.14% (표본) | 2021 오탐은 줄었지만 구분력이 떨어져 사용하지 않음 |
| **WavLM 3층** | 0~5층 비교 후 3층 선택 | **14.10%** | 약한 공격(A07 0.76→0.89, A15 0.70→0.78) 크게 개선 |

### WavLM 층 비교 (2019 eval 8,000개 표본)

| 층 | 0 | 1 | 2 | **3** | 4 | 5 |
|---|---|---|---|---|---|---|
| EER | 32.16% | 24.38% | 18.02% | **14.26%** | 18.11% | 20.15% |
| AUROC | 0.745 | 0.839 | 0.901 | **0.932** | 0.894 | 0.887 |

- 층마다 디퓨전을 새로 학습해 같은 파일로 비교
- 앞쪽 층(0-1)은 A07·A10·A15, 뒤쪽 층(4-5)은 A11·A14를 놓침
- dev real의 val loss는 0층이 가장 낮아 층 선택 기준으로 쓸 수 없음

## 한계와 다음 단계

- **녹음 환경 변화**: 스튜디오 음성으로만 학습해, 전화 코덱을 거친 진짜 음성을 가짜로 보는 경우가 많음 (2021 오탐률 50.4%, 전화 코덱만 보면 75.4%)
- **무음 의존**: 앞뒤 무음을 잘라 낸 2021 음성(hidden)에서는 AUROC가 0.69로 떨어짐. 무음 프레임은 빼지만, WavLM이 파일 전체 문맥을 보기 때문에 무음 정보가 일부 남는 것으로 추정
- **층 선택**: WavLM 3층은 eval 일부(8,000개)를 보고 고름. 고르지 않은 나머지 63,237개에서도 EER이 같았음 (14.26% vs 14.08%)
- **다음 단계**: 구간별 오차 시각화와 의심 구간 재생(XAI), 웹 보고서, 코덱별 임계값
