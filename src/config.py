# ===============================
# config.py
# 모든 경로와 하이퍼파라미터를 한곳에 모아 둔 파일
# preprocess.py, train.py, evaluate.py, explain.py, server.py가 이 파일의 값을 가져다 쓴다.
# 실험할 때는 이 파일의 값만 바꾼다.
#
# 전체 흐름
#   preprocess.py : 음성 → 앞뒤 무음 자름·크기 맞춤 → (증강) → XLS-R 22층 → 프레임 평균 = 파일 벡터(1024) 저장
#   train.py      : train real 벡터만으로 MLP 디퓨전(RDM) 학습
#   evaluate.py   : 노이즈 예측 오차 → dev real 기준 점수·임계값 → eval 지표 + 시험지 판정
# ===============================

import os  # 폴더 경로를 합칠 때 사용한다.

import torch  # GPU 사용 가능 여부를 확인하고 난수 시작값을 고정할 때 사용한다.


# ===============================
# 1. 데이터
# ===============================

DATASET_ROOT = r"H:\Workspace\Dataset"  # 모든 데이터셋이 들어 있는 폴더
LA2019_ROOT = os.path.join(DATASET_ROOT, "LA")  # ASVspoof 2019 LA (train, dev, eval)

# 라벨 파일 한 줄에서 몇 번째 칸(0부터 센다)에 무엇이 있는지
COLUMNS_2019 = {"speaker_id": 0, "file_id": 1, "attack_id": 3, "label": 4}  # 예: "LA_0079 LA_T_1138215 - - bonafide"

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
    # ASVspoof 5 (2024): 최신 TTS·음성 변환 공격 16종(A17~A32). 평가에만 쓴다. eval 음성 중 일부(12,159개)만 내려받아 두었다.
    # 라벨 한 줄: "E_1607 E_0009538969 M C05 2 E_0009486171 AC1 A26 spoof -" (codec이 "-"면 코덱을 거치지 않은 깨끗한 파일)
    "5_eval": {
        "protocol": os.path.join(DATASET_ROOT, "ASVspoof5", "ASVspoof5.eval.track_1.tsv"),
        "audio_dir": os.path.join(DATASET_ROOT, "ASVspoof5", "flac_E_eval"),
        "columns": {"speaker_id": 0, "file_id": 1, "codec": 3, "attack_id": 7, "label": 8},
    },
    # LibriSpeech: 영어 오디오북 낭독 (LibriVox). 스튜디오(2019)와 다른 녹음 조건의 real을 더한다. 라벨 파일 없이 폴더째 모두 real로 읽는다.
    # Hugging Face(openslr/librispeech_asr)에서 받아 화자/챕터 폴더에 flac으로 풀었다. dev-clean과 test-clean은 화자가 겹치지 않는다. (각 40명)
    # 주의: ASVspoof 5의 real도 LibriVox 오디오북(MLS)이라 같은 계열이다. ASVspoof 5 오탐이 줄면 "녹음 조건을 맞춘 효과"라고 밝힌다.
    # "speakers"에는 폴더 이름의 일부를 적는다. 폴더 이름이 이 중 하나를 포함하면 그 데이터에 넣는다. (""는 모든 폴더)
    "libri_train": {"audio_dir": os.path.join(DATASET_ROOT, "LibriSpeech", "dev-clean"), "speakers": [""]},  # 2,703문장
    "libri_dev": {"audio_dir": os.path.join(DATASET_ROOT, "LibriSpeech", "test-clean"), "speakers": [""]},  # 2,620문장
}

# 모델 두 개. 아래 네 줄만 바꿔서 고른다. (학습 seed 3개 평균 EER은 docs/model_explanation.md)
#   논문용: RUN_NAME = "rdm_xlsr_l22",          TRAIN_DATA = ["2019_train"], DEV_DATA = ["2019_dev"], AUGMENT = False  (2019 EER 11.49%)
#   시연용: RUN_NAME = "rdm_xlsr_l22_en_libri", 아래 값 (2019 EER 13.01%, 방 환경 23.92%, ASVspoof 5 17.24%)
RUN_NAME = "rdm_xlsr_l22_en_libri"  # 결과 폴더 이름 (outputs/<RUN_NAME>/)
TRAIN_DATA = ["2019_train", "libri_train"]  # 학습할 real (영어 스튜디오 + 영어 오디오북). bona fide만 쓴다.
DEV_DATA = ["2019_dev", "libri_dev"]  # 조기 종료·보정·임계값용 real (학습에 안 쓴 화자들, 깨끗한 그대로)
AUGMENT = True  # True면 학습 real마다 증강본 하나를 더한다. (방 환경 또는 배속 중 무작위 하나)

EVAL_DATA = "2019_eval"  # 최종 평가에 쓸 데이터 (여기서 처음으로 spoof를 본다)
EVAL_LIMIT = 8000  # eval에서 무작위로 이 개수만 평가한다. 지금까지 비교에 쓴 표본과 같다. (None = 전체)
EXTRA_EVAL = ["5_eval"]  # EVAL_DATA 말고 더 평가할 데이터. 내려받아 둔 파일 중 코덱을 거치지 않은 깨끗한 것만 쓴다. (최신 공격 확인용)


# ===============================
# 2. 결과 폴더
# ===============================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src 폴더의 한 칸 위 = 프로젝트 폴더
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")  # 어디서 실행하든 항상 프로젝트 폴더의 outputs에 저장한다.

# 시험지 폴더: 라벨 파일 없이 모아 둔 음성 (최신 TTS, 내 목소리, 전화 코덱 예시 등). evaluate.py가 끝에 파일마다 점수와 판정을 낸다.
# 정답은 파일 이름으로만 적어 둔다(예: real_..., voice_preview_...는 상용 TTS). 모델·임계값은 이 폴더를 전혀 보지 않는다.
TEST_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "test_samples")

FEATURE_DIR = os.path.join(OUTPUT_DIR, "rdm_features_xlsr_l22")  # 파일 벡터. 데이터·증강마다 한 번만 뽑아 두고 여러 모델이 나눠 쓴다.
RUN_DIR = os.path.join(OUTPUT_DIR, RUN_NAME)  # 이번 모델의 결과 폴더
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")  # 학습한 모델 (train.py)
EVAL_DIR = os.path.join(RUN_DIR, f"evaluation_{EVAL_DATA}")  # 평가 결과 (evaluate.py)


# ===============================
# 3. 음성 읽기
# ===============================

SAMPLE_RATE = 16000  # 모든 음성을 16kHz 모노로 읽는다. (ASVspoof는 원래 16kHz)

# 음성을 읽을 때 항상 하는 두 가지 (스위치 없음. 지난 실험에서 얻은 교훈)
#   1) 앞뒤 무음 자르기: ASVspoof 2019는 앞뒤 무음 길이가 real/fake와 관련이 있다.
#      무음을 그대로 두면 모델이 "합성 흔적" 대신 "무음 길이"를 배운다. (WavLM 3층: 무음 그대로 EER 14% → 잘라 내면 25.6%)
#   2) 최대 진폭 1.0으로 맞추기: 소리 전체에 같은 숫자를 곱할 뿐이라 음색은 그대로다.
#      ASVspoof는 이미 모두 1.0이라 결과가 바뀌지 않고, 작게 녹음된 음성이 "크기만으로" 낯설어 보이는 것을 막는다.
SILENCE_DB_THRESHOLD = -40.0  # 가장 큰 소리보다 이만큼(dB) 작으면 무음으로 보고 앞뒤에서 잘라 낸다. (가운데 쉼은 그대로)
WIN_LENGTH = 400  # 무음을 판단할 때 소리 크기를 재는 창 (400샘플 = 25ms). 12.5ms씩 옮긴다.

# 무음 파일 거르기: 원래 최대 진폭이 이보다 작으면(-60dBFS) "소리가 없는 파일"로 보고 오류를 낸다.
# 크기 맞추기(×1/최대 진폭)를 하면 마이크가 꺼진 녹음의 아주 작은 잡음이 크게 부풀어 "음성"처럼 들어가기 때문이다.
MIN_PEAK = 0.001


# ===============================
# 4. 증강 (학습 real에만. 평가에서는 eval에 방 환경을 입혀 "방에서 녹음한 소리"에서도 버티는지 본다)
# ===============================

# real-only 모델은 처음 보는 녹음 조건을 모두 "낯설다 = fake"로 본다. (예전: 방 환경 real 오탐 96% → 증강 뒤 2%)
# 실제로 녹음할 환경(조용한 방 + 폰·노트북 마이크)에 있는 두 가지만 가볍게 입힌다. 둘 다 코드로 직접 만든다.
#   1) 방 울림: 벽에 반사된 소리가 조금 늦게 섞인다. (2019 real은 스튜디오 녹음이라 울림이 거의 없다)
#   2) 핑크 노이즈: 방·마이크의 바닥 잡음. 저음이 크고 고음으로 갈수록 약하다. (고음은 합성 흔적이 많은 곳이라 화이트 노이즈는 안 쓴다)
ROOM_RT60_RANGE = (0.15, 0.6)  # 울림 길이(초): 소리가 60dB 작아질 때까지 걸리는 시간. 0.15 = 작은 방, 0.6 = 거실 정도
ROOM_DRR_RANGE = (5.0, 15.0)  # 직접 들어온 소리와 울림의 크기 차이(dB). 클수록 마이크가 입에 가깝다.
ROOM_SNR_RANGE = (25.0, 40.0)  # 말소리와 바닥 잡음의 크기 차이(dB). 배경에 살짝 깔리는 정도
ROOM_NOISE_LOW_HZ = 50.0  # 핑크 노이즈에서 이보다 낮은 소리는 뺀다. (귀에 안 들리는 초저음에 힘이 몰려서)
ROOM_PAD_SEC_RANGE = (0.0, 1.0)  # 앞뒤에 붙이는 "잡음만 있는 구간" 길이(초). 실제 녹음은 말 앞뒤에 방 잡음이 깔린다.
# 배속 증강: 말 속도·목소리 높이를 함께 바꾼다. (테이프를 빨리 감는 것과 같다) 목소리가 높거나 빠른 real도 real로 보게
AUGMENT_SPEEDS = [0.9, 1.1, 1.2]  # 0.9 = 약 2음 낮게, 1.1 = 약 2음 높게, 1.2 = 약 3음 높게


# ===============================
# 5. XLS-R (특징 추출기. 학습하지 않고 숫자로 바꾸는 일만 한다)
# ===============================

# 음성 파일 하나 → XLS-R 22층 프레임 특징 → 프레임 평균 = 벡터 하나(1024). 그 벡터들 위에서 작은 MLP 디퓨전을 real만으로 학습한다.
# 판정은 디퓨전의 노이즈 예측 오차로 한다.
# 프레임마다 디퓨전을 돌리면 오차가 "그 순간 소리가 얼마나 복잡한가"에 흔들렸다(EER 20%). 파일 평균으로 그 흔들림을 없앤다. (EER 11%)
XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"  # 사전학습 음성 모델 (128개 언어, 약 3억 가중치). 처음 실행 때 내려받는다.
XLSR_LAYER = 22  # 24층 중 몇 번째 층을 쓸지. 19~24층 중 22층 (층 후보는 eval을 본 비교에서 나왔다 → 발표 때 밝힌다)
XLSR_DIM = 1024  # XLS-R 한 층의 특징 차원 = 파일 벡터 크기 = 디퓨전 입력 크기
XLSR_FRAME_SEC = 0.02  # XLS-R 프레임 하나 = 320샘플 = 20ms (모델 구조로 정해진 값)


# ===============================
# 6. 디퓨전 (DDPM)
# ===============================

TIMESTEPS = 1000  # 노이즈를 섞는 전체 단계 수
BETA_START = 0.0001  # 첫 단계에서 추가할 노이즈의 크기
BETA_END = 0.02  # 마지막 단계에서 추가할 노이즈의 크기 (선형 스케줄)
TIME_EMBEDDING_DIM = 128  # timestep 숫자 하나를 128개의 특징값으로 바꾼다.


# ===============================
# 7. MLP와 학습
# ===============================

HIDDEN = 1024  # MLP 안쪽 크기 (처음부터 끝까지 폭이 같다)
BLOCKS = 3  # MLP 잔차 블록 수
DROPOUT = 0.1  # real 표본이 적어서 외우지 않도록 일부 값을 무작위로 끈다.
BATCH_SIZE = 256  # 벡터는 작아서 한 번에 많이 넣는다.
LEARNING_RATE = 0.0001  # 가중치를 한 번에 수정하는 정도 (AdamW)
STEPS = 40000  # 학습 걸음 수 (걸음 하나 = 256개 묶음 하나). 데이터 양이 달라도 학습 길이가 같도록 epoch 대신 걸음으로 센다.
VAL_EVERY = 500  # 이 걸음마다 dev real 손실을 재고, 가장 낮았던 모델을 남긴다.
GRAD_CLIP = 1.0  # gradient가 갑자기 커지는 것을 막는 상한
VAL_SEED = 1234  # 검증용 t와 노이즈를 뽑는 전용 난수 시작값 (학습 난수와 섞지 않는다)


# ===============================
# 8. 점수와 판정
# ===============================

# 점수에 쓸 t. 지난 실험(Mel, XLS-R) 모두 t=500 이상은 real/fake를 거의 못 갈랐고 50~200이 가장 좋았다.
# 그래서 이 모델의 결과를 보기 전에 50/100/200으로 정해 두었다.
SCORE_TIMESTEPS = [50, 100, 200]
NOISE_COUNT = 16  # 점수를 낼 때 t마다 고정 노이즈 16개로 시험을 보고 평균한다. (벡터 하나라 싸다)

THRESHOLD_PERCENTILE = 90  # dev bona fide 파일 점수의 90백분위를 임계값으로 쓴다. = real 10%까지는 fake로 의심해도 감수한다.
# 처음엔 관례대로 95(오탐 5%)였다. "fake를 놓치는 것이 real을 한 번 더 의심하는 것보다 나쁘다"는 사용 목적으로 90으로 바꿨다. (2026-11-25, eval 탐지율을 본 뒤라 발표 때 밝힌다)


# ===============================
# 9. 웹 데모 (server.py, explain.py)
# ===============================

WEB_DIR = os.path.join(PROJECT_ROOT, "web")  # index.html이 있는 폴더
MAX_UPLOAD_MB = 20  # 업로드 크기 상한
MAX_AUDIO_SEC = 30.0  # 이보다 긴 음성은 앞부분만 분석한다. (GPU 메모리, 응답 시간)
MIN_AUDIO_SEC = 1.0  # 앞뒤 무음을 뺀 말소리가 이보다 짧으면 분석하지 않는다.
SERVER_HOST = "127.0.0.1"  # 내 컴퓨터에서만 접속 가능. "0.0.0.0"으로 열면 음성이 네트워크로 오가므로 HTTPS가 필요하다.
SERVER_PORT = 8000  # 브라우저 주소: http://127.0.0.1:8000
# 웹 설명(explain.py): 판정은 파일 전체 벡터 하나로 한다. 아래 창은 "어디가 수상한지" 보여 주는 설명용이다.
EXPLAIN_WINDOW_SEC = 2.0  # 설명용 창 길이: 이 길이의 프레임만 평균 낸 벡터로 RDM 점수를 따로 낸다.
EXPLAIN_HOP_SEC = 1.0  # 창을 옮기는 간격
EXPLAIN_REF_FILES = 1000  # 창 점수 기준(평균·표준편차·임계값)을 만들 dev real 파일 수 (무작위, SEED 고정). XLS-R을 다시 돌려야 해서 일부만 쓴다.
DIST_TIMESTEPS = [50, 200]  # 웹의 "분포 거리"(참고용)에 쓸 t 두 개. 이 두 오차(z)의 조합이 dev real 무리에서 얼마나 떨어졌는지 (마할라노비스). 판정에는 쓰지 않는다.
DISPLAY_POINTS = 1500  # 웹 산점도에 찍을 점 수 (real, fake 각각). 가짜는 EVAL_DATA의 spoof로, 화면 비교용으로만 쓴다.
DISPLAY_N_MELS = 64  # 화면용 Mel 그림의 세로 칸 수 (모델 입력 아님)
N_FFT = 512  # 화면용 Mel 그림의 FFT 크기 (모델 입력 아님)


# ===============================
# 10. 실행 장치와 난수
# ===============================

SEED = 42  # 실험을 반복해도 같은 결과가 나오도록 난수 시작값을 고정한다. (eval 표본 고르기, 고정 노이즈에도 쓴다)

if torch.cuda.is_available():  # NVIDIA GPU를 사용할 수 있으면
    DEVICE = torch.device("cuda")  # GPU로 계산한다.
else:
    DEVICE = torch.device("cpu")  # 아니면 CPU로 계산한다.

torch.manual_seed(SEED)  # PyTorch 난수 시작값을 고정한다.
