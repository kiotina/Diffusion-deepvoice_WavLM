# ===============================
# config.py
# 모든 경로와 하이퍼파라미터를 한곳에 모아 둔 파일
# preprocess.py, train.py, evaluate.py가 이 파일의 값을 가져다 쓴다.
# 실험할 때는 이 파일의 값만 바꾼다.
# ===============================

import os  # 폴더 경로를 합치고 폴더를 만들 때 사용한다.

import torch  # GPU 사용 가능 여부를 확인하고 난수 시작값을 고정할 때 사용한다.


# ===============================
# 1. 경로
# ===============================

DATASET_ROOT = r"H:\Workspace\Dataset"  # 모든 데이터셋이 들어 있는 폴더
LA2019_ROOT = os.path.join(DATASET_ROOT, "LA")  # ASVspoof 2019 LA (train, dev, eval)
LA2021_ROOT = os.path.join(DATASET_ROOT, "ASVspoof2021_LA_eval")  # ASVspoof 2021 LA (eval만 있다)
KO_ROOT = os.path.join(DATASET_ROOT, "원천데이터")  # AIHub 한국어 음성 (모두 진짜 사람 음성. 화자마다 폴더 하나)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src 폴더의 한 칸 위 = 프로젝트 폴더
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")  # 어디서 실행하든 항상 프로젝트 폴더의 outputs에 저장한다.

# 라벨 파일 한 줄에서 몇 번째 칸(0부터 센다)에 무엇이 있는지
COLUMNS_2019 = {"speaker_id": 0, "file_id": 1, "attack_id": 3, "label": 4}  # 예: "LA_0079 LA_T_1138215 - - bonafide"
COLUMNS_2021 = {"speaker_id": 0, "file_id": 1, "attack_id": 4, "label": 5}  # 예: "LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval"

DATASETS = {  # 데이터 이름 → 라벨 파일, 음성 폴더, 칸 번호. 새 데이터셋은 여기에 한 항목을 추가한다.
    "2019_train": {
        "protocol": os.path.join(LA2019_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"),
        "audio_dir": os.path.join(LA2019_ROOT, "ASVspoof2019_LA_train", "flac"),
        "columns": COLUMNS_2019,
    },
    "2019_dev": {
        "protocol": os.path.join(LA2019_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"),
        "audio_dir": os.path.join(LA2019_ROOT, "ASVspoof2019_LA_dev", "flac"),
        "columns": COLUMNS_2019,
    },
    "2019_eval": {
        "protocol": os.path.join(LA2019_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt"),
        "audio_dir": os.path.join(LA2019_ROOT, "ASVspoof2019_LA_eval", "flac"),
        "columns": COLUMNS_2019,
    },
    "2021_eval": {
        # 폴더 안의 ASVspoof2021.LA.cm.eval.trl.txt에는 파일ID만 있고 정답(real/fake)이 없다.
        # 정답은 따로 공개된 키 파일(LA-keys-full.tar.gz)에 있다. LA2021_ROOT 안에 풀어 둔 keys 폴더를 쓴다.
        "protocol": os.path.join(LA2021_ROOT, "keys", "LA", "CM", "trial_metadata.txt"),
        "audio_dir": os.path.join(LA2021_ROOT, "flac"),
        "columns": COLUMNS_2021,
    },
    # 한국어 데이터에는 라벨 파일이 없다. 대신 폴더를 뒤져 wav를 모두 찾고(전부 real), 폴더 이름(= 화자)으로 train/dev를 나눈다.
    # "speakers"에는 화자 폴더 이름의 일부를 적는다. 폴더 이름이 이 중 하나를 포함하면 그 분할에 넣는다.
    # 화자가 5명뿐이라 dev는 1명이다. real-only 원칙상 임계값은 학습에 안 쓴 화자로 정해야 해서, 1명이라도 따로 뺀다.
    "ko_train": {
        "audio_dir": KO_ROOT,
        "speakers": [
            "KYG",  # 녹음 부스, TTS 학습용 데이터 (93분, 원래 48kHz → 읽을 때 16kHz로 바꾼다)
            "1526752256",  # 자유대화, 실내, 남성 (55분)
            "1527404971",  # 자유대화, 실내, 남성 (65분)
            "1524374974",  # 자유대화, 실내, 여성 (52분)
        ],
    },
    "ko_dev": {
        "audio_dir": KO_ROOT,
        "speakers": [
            "1536505292",  # 자유대화, 실내, 여성 (60분). 웹에 들어올 음성(일상 녹음)과 비슷한 조건이라 dev로 둔다.
        ],
    },
}

# 영어 모델: TRAIN_DATA = "2019_train", DEV_DATA = "2019_dev", RUN_NAME = "wavlm_l3_trim", TRIM_SILENCE = True (무음 자른 것이 새 기준선. 무음 그대로인 예전 최종 모델은 "wavlm_l3", TRIM_SILENCE = False)
# 한국어 모델: TRAIN_DATA = "ko_train", DEV_DATA = "ko_dev", RUN_NAME = "wavlm_l3_ko" (한국어 fake가 없어서 eval은 아직 못 한다)
TRAIN_DATA = "2019_train"  # 학습에 쓸 데이터 (bona fide만 사용)
DEV_DATA = "2019_dev"  # 검증·보정·임계값에 쓸 데이터 (bona fide만 사용)
EVAL_DATA = "2019_eval"  # 최종 평가에 쓸 데이터. 평가 데이터를 바꿀 때는 이 이름만 바꾼다.

RUN_NAME = "wavlm_l3_trim"  # 실험 이름. 결과가 outputs/<RUN_NAME>/에 따로 저장된다. 설정을 바꿔 새로 실험할 때는 이름도 바꾼다. (첫 실험: "baseline")
RUN_DIR = os.path.join(OUTPUT_DIR, RUN_NAME)  # 이번 실험의 결과 폴더

PREPROCESS_DIR = os.path.join(RUN_DIR, "preprocess")  # 전처리 결과를 저장할 폴더
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")  # 학습한 모델을 저장할 폴더
EVAL_DIR = os.path.join(RUN_DIR, f"evaluation_{EVAL_DATA}")  # 평가 결과를 저장할 폴더 (평가 데이터마다 따로 저장)


# ===============================
# 2. 음성 설정
# ===============================

SAMPLE_RATE = 16000  # 모든 음성을 16kHz로 사용한다. (ASVspoof는 원래 16kHz)
SEGMENT_SEC = 2.0  # 모델에 넣는 조각 하나의 길이(초)
MIN_SEGMENT_SEC = 1.0  # 이보다 짧게 남은 마지막 조각은 버린다. (검증·평가용 조각)


# ===============================
# 3. log-Mel 설정
# ===============================

N_FFT = 512  # 한 번에 FFT를 계산할 샘플 수
WIN_LENGTH = 400  # 실제로 보는 창의 길이 (400샘플 = 25ms)
HOP_LENGTH = 256  # 창을 옮기는 간격 (256샘플 = 16ms)
N_MELS = 80  # Mel 주파수 칸 수 (그림의 세로 칸 수)
LOG_EPS = 1e-6  # log(0)을 막기 위해 더하는 아주 작은 값

# 디퓨전에 넣을 입력의 종류
#   "mel"   : log-Mel (N_MELS칸). 처음 방식
#   "spec"  : Mel 필터를 거치지 않은 log 스펙트로그램 (256칸). Mel은 고음일수록 넓은 범위를 한 칸으로 합쳐서, 고음에 남는 합성 흔적이 뭉개진다.
#   "wavlm" : WavLM이 음성에서 뽑은 특징 (768차원). 아래 "WavLM 설정" 참고
FEATURE_TYPE = "wavlm"


# ===============================
# 3-1. WavLM 설정 (FEATURE_TYPE = "wavlm"일 때만 사용)
# ===============================

# WavLM: 진짜 사람 음성 9만 4천 시간으로 미리 학습된 음성 모델 (fake는 본 적 없다 → real-only 원칙과 맞는다).
# 음성을 넣으면 20ms마다 768개 숫자로 된 특징을 층(layer)마다 내놓는다. 우리는 그중 한 층만 꺼내 디퓨전의 입력으로 쓴다.
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"  # Hugging Face에서 받는 모델 이름 (처음 실행할 때 한 번 내려받는다)
WAVLM_LAYER = 3  # 몇 번째 층의 특징을 쓸지 (0 = CNN 출력, 1~12 = Transformer 층). 앞쪽 층일수록 "소리의 결", 뒤쪽일수록 "말의 내용"을 담는다.
# 주의: 3층은 0~5층으로 디퓨전을 각각 학습하고 eval 8,000개 표본(seed 42)의 점수를 비교해 고른 값이다. (EER 0층 32% → 3층 14% → 5층 20%)
#       고를 때 쓰지 않은 나머지 63,237개에서도 EER이 같아(14.26% vs 14.08%) 표본에 맞춘 효과는 없었지만, 보고할 때 이 점을 밝힌다.
WAVLM_DIM = 768  # 특징 하나의 길이 (wavlm-base-plus는 768)
WAVLM_HOP = 320  # WavLM 특징 한 프레임 = 320샘플 = 20ms (모델 구조로 정해진 값)

FRAME_HOP = WAVLM_HOP if FEATURE_TYPE == "wavlm" else HOP_LENGTH  # 프레임 하나가 몇 샘플씩 떨어져 있는지 (Mel·스펙트로그램 256, WavLM 320)
N_BINS = {"mel": N_MELS, "spec": N_FFT // 2, "wavlm": WAVLM_DIM}[FEATURE_TYPE]  # 모델에 들어가는 세로 칸 수. 스펙트로그램은 257칸 중 맨 위 한 칸(8kHz)을 버리고 256칸을 쓴다. (2D U-Net은 두 번 반으로 줄이므로 4로 나눠떨어져야 한다)

SEGMENT_FRAMES = 1 + int(SEGMENT_SEC * SAMPLE_RATE) // FRAME_HOP  # 2초 조각의 프레임 수 (Mel 126, WavLM 101)
MIN_SEGMENT_FRAMES = 1 + int(MIN_SEGMENT_SEC * SAMPLE_RATE) // FRAME_HOP  # 1초 조각의 프레임 수 (Mel 63, WavLM 51)


# ===============================
# 4. 무음 mask 설정 (기능만 만들어 두고 기본값은 끈다)
# ===============================

MASK_SILENCE = True  # True면 무음 프레임도 mask=0으로 만들어 loss와 점수에서 뺀다.
SILENCE_DB_THRESHOLD = -40.0  # 파일에서 가장 큰 프레임보다 이만큼(dB) 작으면 무음으로 본다.

# 무음 자르기 실험: 음성을 읽자마자 앞뒤 무음을 파형에서 잘라 낸다. (기준은 위의 -40dB와 같다)
# mask는 무음 프레임을 점수에서 뺄 뿐이고, WavLM은 파일 전체를 보고 특징을 만들기 때문에 무음 정보가 말소리 프레임에 샐 수 있다.
# 잘라 내면 WavLM이 앞뒤 무음을 아예 못 본다. 그래도 real/fake를 가르면 "무음 길이만 배운 것은 아니다"라는 근거가 된다.
# 주의: 켠 채로 preprocess.py를 돌리면 학습 데이터도 잘린다. 평가 결과는 evaluation_<데이터>_trim 폴더에 따로 저장된다.
TRIM_SILENCE = True
EVAL_DIR += "_trim" if TRIM_SILENCE else ""  # 무음을 잘라 낸 평가는 원래 결과를 덮어쓰지 않게 따로 저장한다.


# ===============================
# 4-1. 코덱 증강 (학습 데이터만. 진짜 음성을 실제 전화 코덱에 통과시킬 뿐 fake는 보지 않는다 → real-only 유지)
# ===============================

# 2019 real은 모두 스튜디오 녹음이라, 모델이 전화망을 거친 진짜 음성을 "처음 듣는 소리 = 가짜"로 판단했다. (2021 eval 오탐률 69~97%)
# 그래서 train real마다 원본 1개 + 아래 코덱 중 무작위 1개를 같이 학습해 "전화 소리도 진짜일 수 있다"를 가르친다.
# 코덱은 ffmpeg(imageio-ffmpeg, venv 안에 설치)로 실제로 인코딩했다가 다시 디코딩한다.
# 2021의 gsm, opus는 일부러 학습에 넣지 않는다. 처음 보는 코덱에서도 좋아지는지 확인하는 시험으로 남긴다.
# (ulaw는 alaw와 같은 표준 G.711이고, pstn 흉내의 마지막 단계에도 쓰이므로 "거의 본 코덱"으로 따로 본다)
AUGMENT_CODEC = False  # True면 위처럼 증강한다. False면 원래대로 원본만 학습한다.
AUGMENT_CODECS = ["alaw", "g722", "pstn"]  # 학습에 쓰는 코덱. alaw: 전화(8kHz), g722: 인터넷 전화(16kHz), pstn: 일반 전화망 흉내
CODEC_SETTINGS = {  # ffmpeg 출력 형식 이름과 코덱이 쓰는 샘플링 주파수. 두 코덱 모두 표준이라 비트레이트를 고를 수 없다. (64 kbps 고정)
    "alaw": ("alaw", 8000),  # G.711 A-law: 8kHz로 줄이고 8비트로 압축 → 4kHz 위가 사라진다.
    "mulaw": ("mulaw", 8000),  # G.711 μ-law: A-law와 같은 표준의 다른 압축 곡선 (pstn 흉내의 마지막 단계에만 쓴다)
    "g722": ("g722", 16000),  # G.722: 16kHz 그대로, 대역을 둘로 나눠 압축 → 7kHz 근처까지 남는다.
}
# pstn 흉내: 실제 2021 pstn은 "스마트폰 → 일반 전화망(코덱 모름) → μ-law로 녹음"이다. 일반 전화망이 통과시키는 대역(약 300~3400Hz)만 남긴 뒤 μ-law에 통과시킨다.
# 대역 경계를 하나로 못 박지 않고 파일마다 아래 범위에서 무작위로 고른다. (특정 설정만 외우지 않도록)
PSTN_LOW_HZ_RANGE = (200, 400)  # 낮은 쪽 경계 (Hz)
PSTN_HIGH_HZ_RANGE = (3000, 3600)  # 높은 쪽 경계 (Hz)


# ===============================
# 4-2. 환경음 증강 (학습 데이터만. 진짜 음성 + 진짜 환경음 → real-only 유지)
# ===============================

# 2019 real은 모두 조용한 스튜디오 녹음이라, 잡음 섞인 진짜 음성을 "낯설다 = 가짜"로 판단할 수 있다.
# 그래서 train real마다 원본 1개 + 환경음을 섞은 것 1개를 같이 학습해 "잡음이 섞여도 진짜일 수 있다"를 가르친다.
# 환경음: ESC-50 (환경음 50종 × 40개, 5초, CC BY-NC 3.0. 출처: github.com/karolpiczak/ESC-50)
NOISE_ROOT = os.path.join(DATASET_ROOT, "ESC-50")  # audio 폴더와 meta/esc50.csv가 있다.
AUGMENT_NOISE = False  # True면 위처럼 증강한다.
NOISE_EXCLUDE = ["crying_baby", "laughing", "coughing", "sneezing", "breathing", "snoring"]  # 목소리 성분이 강한 6종은 뺀다. ("다른 사람 목소리"를 진짜로 배우지 않도록) → 44종 사용
NOISE_TRAIN_FOLDS = [1, 2, 3, 4]  # 학습 증강에 쓰는 묶음 (1,600개 중 44종 = 1,408개)
NOISE_EVAL_FOLDS = [5]  # 평가용 잡음 묶음. 학습 때 들어 본 적 없는 잡음으로 시험한다. (352개)
NOISE_SNR_RANGE = (5.0, 20.0)  # 학습 때 말소리와 잡음의 크기 차이(dB)를 이 범위에서 무작위로 고른다. 작을수록 잡음이 크다.

# 앞뒤 빈 구간: 실제 잡음 환경 녹음은 말 앞뒤에 "잡음만 있는 구간"이 있다. (잡음이 -40dB 기준보다 커서 무음 자르기로 안 잘린다)
# 그래서 무음을 자른 말소리 앞뒤에 빈 구간을 붙이고 → 그 위에 잡음을 깐다. 빈 구간은 반드시 잡음과 함께만 쓴다.
# 길이는 파일마다 이 범위에서 무작위로 고르고, real·fake 모두에 같은 규칙을 쓴다. → 길이가 real/fake 단서가 될 수 없다.
NOISE_PAD_SEC_RANGE = (0.0, 1.0)  # 앞, 뒤 각각 (초)

# 잡음 평가: 숫자를 넣으면 eval 파일(real·fake 모두)에 평가용 잡음을 그 SNR로 섞어서 평가한다. None이면 깨끗한 그대로.
# dev real은 깨끗한 그대로 둔다. → "깨끗한 real로 정한 임계값이 잡음 환경에서 얼마나 오탐을 내는가"를 잰다.
# 같은 파일은 언제 돌려도 같은 잡음·SNR·빈 구간이 섞인다. (파일 ID로 난수를 정한다)
EVAL_NOISE_SNR = None  # 예: 20, 10, 5
EVAL_DIR += f"_noise{EVAL_NOISE_SNR}" if EVAL_NOISE_SNR is not None else ""  # 잡음 평가는 SNR마다 결과를 따로 저장한다. (예: evaluation_2019_eval_trim_noise10)


# ===============================
# 5. Diffusion 설정
# ===============================

TIMESTEPS = 1000  # 노이즈를 섞는 전체 단계 수
BETA_START = 0.0001  # 첫 단계에서 추가할 노이즈의 크기
BETA_END = 0.02  # 마지막 단계에서 추가할 노이즈의 크기 (선형 스케줄)


# ===============================
# 6. 모델 설정
# ===============================

BASE_CHANNELS = 32  # U-Net 첫 단계의 채널 수 (32 → 64 → 128)
TIME_EMBEDDING_DIM = 128  # timestep 숫자 하나를 128개의 특징값으로 바꾼다.
GROUP_NORM_GROUPS = 8  # GroupNorm에서 채널을 몇 묶음으로 나눠 정규화할지 (32, 64, 128 모두 나눠떨어진다)
UNET_FRAMES = 4 * ((SEGMENT_FRAMES + 3) // 4)  # U-Net 안에서 프레임 수를 4의 배수로 채운다. (2번 반으로 줄여도 딱 나눠지도록. Mel 126 → 128, WavLM 101 → 104)
UNET1D_CHANNELS = 256  # WavLM용 1D U-Net 첫 단계의 채널 수 (256 → 512 → 512). 입력이 768채널이라 2D U-Net(32)보다 크게 잡는다.


# ===============================
# 7. 학습 설정
# ===============================

BATCH_SIZE = 32  # 한 번에 학습할 조각 수 (RTX 3070 Ti 8GB 기준으로 시작)
LEARNING_RATE = 0.0001  # 가중치를 한 번에 수정하는 정도
MAX_EPOCHS = 100  # 최대 학습 횟수
PATIENCE = 10  # 이 횟수만큼 val loss가 좋아지지 않으면 학습을 멈춘다. (early stopping)
GRAD_CLIP = 1.0  # gradient가 갑자기 커지는 것을 막는 상한
SAVE_EVERY = 10  # 이 epoch마다 체크포인트를 따로 저장한다. (나중에 분석용)
NUM_WORKERS = 0  # 데이터를 불러오는 별도 프로세스 수 (Windows에서는 0이 안전하다)

SEED = 42  # 실험을 반복해도 같은 결과가 나오도록 난수 시작값을 고정한다.
VAL_SEED = 1234  # 검증용 t와 노이즈를 뽑는 전용 난수 시작값
VAL_T_BIN_SIZE = 200  # val loss를 t 구간(0–200, 200–400, …, 800–1000)별로 나눠 기록할 때 구간 폭
LOSS_EPS = 1e-8  # masked MSE에서 mask=1 칸이 0개일 때 0으로 나누지 않도록 막는 값

# 학습 실험 스위치 (기본값이면 원래 코드와 똑같이 동작한다)
TRAIN_T_MAX = TIMESTEPS  # 학습·검증에서 뽑는 t의 상한 (기본 1000). 점수에는 t ≤ 500만 쓰므로 600으로 줄여 그 구간에 집중해 볼 수 있다. VAL_T_BIN_SIZE의 배수로 둔다.
USE_EMA = False  # True면 가중치의 이동 평균(EMA) 모델로 검증하고 저장한다. (epoch마다 가중치가 흔들리는 것을 줄인다)
EMA_DECAY = 0.995  # EMA가 예전 가중치를 유지하는 비율. 1 epoch ≈ 81 step(2,580개 ÷ 32)이라 대략 최근 2~3 epoch의 평균이 된다.


# ===============================
# 8. 점수 계산과 판정 설정
# ===============================

SCORE_TIMESTEPS = [100, 300, 500]  # 이상 점수에 사용할 t
SCORE_NOISE_SEEDS = [1729, 2718]  # t마다 사용할 고정 노이즈의 시작값
INFER_HOP_SEC = 1.0  # 평가할 때 2초 창을 1초씩 밀면서 자른다.
INFER_HOP_FRAMES = int(INFER_HOP_SEC * SAMPLE_RATE) // FRAME_HOP  # 1초 = Mel 62프레임, WavLM 50프레임

NUM_BANDS = 1 if FEATURE_TYPE == "wavlm" else 8  # 주파수 칸(N_BINS)을 이 개수의 대역으로 나눠 오차를 따로 잰다. 스펙트로그램 256칸이면 8 → 대역마다 32칸(1kHz씩). (N_BINS를 나눠떨어지게 할 것. Mel 80칸이면 8)
# WavLM의 768차원은 주파수처럼 위아래 순서가 있는 값이 아니라서, 이웃한 차원끼리 묶는 "대역"이 의미가 없다. 그래서 1(전체 하나)로 둔다.

THRESHOLD_PERCENTILE = 95  # dev bona fide 점수의 95백분위를 임계값으로 쓴다.


# ===============================
# 9. 개발용 설정 (작게 돌려볼 때만 사용)
# ===============================

FILE_LIMIT = None  # 숫자를 넣으면 분할마다 앞에서부터 그 개수의 파일만 사용한다. (예: 100)
EVAL_LIMIT = 8000  # 숫자를 넣으면 eval에서 무작위로 그 개수만 평가한다. (예: 2000)


# ===============================
# 10. 실행 장치와 난수
# ===============================

if torch.cuda.is_available():  # NVIDIA GPU를 사용할 수 있는지 확인
    DEVICE = torch.device("cuda")  # GPU를 사용할 수 있으면 GPU로 계산한다.
else:  # 사용할 수 있는 GPU가 없다면
    DEVICE = torch.device("cpu")  # CPU로 계산한다.

torch.manual_seed(SEED)  # PyTorch 난수 시작값을 고정한다.



# ===============================
# 11. 웹 데모 설정 (explain.py, server.py)
# ===============================

EXPLAIN_DIR = os.path.join(RUN_DIR, "explain")  # 설명용 기준값을 저장할 폴더 (실험마다 따로)
REFERENCE_PATH = os.path.join(EXPLAIN_DIR, "reference.json")  # explain.py가 dev real로 만든 기준값
WEB_DIR = os.path.join(PROJECT_ROOT, "web")  # index.html이 있는 폴더

DECISION_KEY = "score_mean"  # 웹에서 판정에 쓸 점수 (score_mean, score_max, score_maha 중 하나). eval 결과가 가장 좋았던 것으로 바꾼다.
MAX_UPLOAD_MB = 20  # 업로드 크기 상한
MAX_AUDIO_SEC = 30.0  # 이보다 긴 음성은 앞부분만 분석한다. (GPU 메모리, 응답 시간)
SCORE_CHUNK = 16  # U-Net에 한 번에 넣을 구간 수 (× t 3개 × 노이즈 2개). 메모리가 부족하면 줄인다.
DISPLAY_N_MELS = 64  # 화면에 깔아 둘 Mel 그림의 칸 수. 모델 입력이 아니라 사람이 위치를 찾는 배경용이다.

SERVER_HOST = "127.0.0.1"  # 내 컴퓨터에서만 접속 가능. "0.0.0.0"으로 열면 음성이 네트워크로 오가므로 HTTPS가 필요하다.
SERVER_PORT = 8000  # 브라우저 주소: http://127.0.0.1:8000