# ===============================
# config.py
# 모든 경로와 하이퍼파라미터를 한곳에 모아 둔 파일
# preprocess.py, train.py, evaluate.py가 이 파일의 값을 가져다 쓴다.
# 실험할 때는 이 파일의 값만 바꾼다.
# ===============================

import os  # 폴더 경로를 합칠 때 사용한다.

import torch  # GPU 사용 가능 여부를 확인하고 난수 시작값을 고정할 때 사용한다.


# ===============================
# 1. 데이터 경로
# ===============================

DATASET_ROOT = r"H:\Workspace\Dataset"  # 모든 데이터셋이 들어 있는 폴더
LA2019_ROOT = os.path.join(DATASET_ROOT, "LA")  # ASVspoof 2019 LA (train, dev, eval)
LA2021_ROOT = os.path.join(DATASET_ROOT, "ASVspoof2021_LA_eval")  # ASVspoof 2021 LA (eval만 있다)
KO_ROOT = os.path.join(DATASET_ROOT, "원천데이터")  # AIHub 한국어 음성 (모두 진짜 사람 음성. 화자마다 폴더 하나)
KSPON_WAV_DIR = os.path.join(DATASET_ROOT, "AI-Hub_한국어 음성", "wav")  # KsponSpeech(약 2,000명 자유 대화)에서 골라 꺼낸 wav (train/, dev/. xlsr-final 태그의 prepare_kspon.py)

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
        # 폴더 안의 ASVspoof2021.LA.cm.eval.trl.txt에는 정답이 없다. 따로 공개된 키 파일(keys 폴더)을 쓴다.
        "protocol": os.path.join(LA2021_ROOT, "keys", "LA", "CM", "trial_metadata.txt"),
        "audio_dir": os.path.join(LA2021_ROOT, "flac"),
        "columns": COLUMNS_2021,
    },
    # 한국어 데이터에는 라벨 파일이 없다. 폴더를 뒤져 wav를 모두 찾고(전부 real), 폴더 이름(= 화자)으로 train/dev를 나눈다.
    # "speakers"에는 화자 폴더 이름의 일부를 적는다. 폴더 이름이 이 중 하나를 포함하면 그 데이터에 넣는다.
    "ko_train": {
        "audio_dir": KO_ROOT,
        "speakers": [
            "KYG",  # 녹음 부스, TTS 학습용 데이터 (원래 48kHz → 읽을 때 16kHz로 바꾼다)
            "1526752256",  # 자유대화, 실내, 남성
            "1527404971",  # 자유대화, 실내, 남성
            "1524374974",  # 자유대화, 실내, 여성
        ],
    },
    "ko_dev": {
        "audio_dir": KO_ROOT,
        "speakers": ["1536505292"],  # 자유대화, 실내, 여성. 학습에 안 쓴 화자로 임계값을 정한다. (real-only)
    },
    "kspon_train": {"audio_dir": os.path.join(KSPON_WAV_DIR, "train"), "speakers": ["KsponSpeech_"]},  # 세션 폴더 = 화자 자리
    "kspon_dev": {"audio_dir": os.path.join(KSPON_WAV_DIR, "dev"), "speakers": ["eval_"]},  # eval_clean, eval_other (train과 다른 화자)
}

TRAIN_DATA = "2019_train"  # 학습에 쓸 데이터 (bona fide만 사용)
DEV_DATA = "2019_dev"  # 검증·보정·임계값에 쓸 데이터 (bona fide만 사용)
EVAL_DATA = "2019_eval"  # 최종 평가에 쓸 데이터 (여기서 처음으로 spoof를 본다)


# ===============================
# 2. 결과 폴더
# ===============================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src 폴더의 한 칸 위 = 프로젝트 폴더
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")  # 어디서 실행하든 항상 프로젝트 폴더의 outputs에 저장한다.

# 시험지 폴더: 라벨 파일 없이 모아 둔 음성 (최신 TTS, 내 목소리, 전화 코덱 예시 등). evaluate.py가 끝에 파일마다 점수와 판정을 낸다.
# 정답은 파일 이름으로만 적어 둔다(예: real_..., voice_preview_...는 상용 TTS). 모델·임계값은 이 폴더를 전혀 보지 않는다.
TEST_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "test_samples")

RUN_NAME = "all4"  # 실험 이름. 특징 목록을 바꾸면 이름도 바꾼다. (예: "phase", "spec_phase") 결과가 outputs/<RUN_NAME>/에 따로 저장된다. 설정을 바꿔 새로 실험할 때는 이름도 바꾼다.
RUN_DIR = os.path.join(OUTPUT_DIR, RUN_NAME)  # 이번 실험의 결과 폴더

PREPROCESS_DIR = os.path.join(RUN_DIR, "preprocess")  # 전처리 결과
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")  # 학습한 모델
EVAL_DIR = os.path.join(RUN_DIR, f"evaluation_{EVAL_DATA}")  # 평가 결과 (평가 데이터마다 따로)


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

# 무음 파일 거르기: 원래 최대 진폭이 이보다 작으면(-60dBFS) "소리가 없는 파일"로 보고 오류를 낸다.
# 크기 맞추기(×1/최대 진폭)를 하면 마이크가 꺼진 녹음의 아주 작은 잡음이 크게 부풀어 "음성"처럼 들어가기 때문이다.
# ASVspoof는 모두 최대 진폭 1.0이라 걸리지 않는다.
MIN_PEAK = 0.001

# 방 환경 증강 (rdm_features.py가 학습 real에 쓴다. 평가에서는 eval에 입혀 "방에서 녹음한 소리"에서도 버티는지 본다)
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
# 4. 입력 특징 (디퓨전이 보는 그림)
# ===============================

N_FFT = 512  # 한 번에 FFT를 계산할 샘플 수 → 주파수 칸 257개 (0 ~ 8000Hz, 31.25Hz 간격)
WIN_LENGTH = 400  # 실제로 보는 창의 길이 (400샘플 = 25ms)
HOP_LENGTH = 256  # 창을 옮기는 간격 (256샘플 = 16ms)
LOG_EPS = 1e-6  # log(0)을 막기 위해 더하는 아주 작은 값

# 디퓨전에 넣을 특징 목록. 여러 개를 넣으면 같은 크기(256칸 × 프레임)의 그림을 채널로 쌓는다. (사진의 R·G·B처럼)
# 모든 특징이 같은 STFT 한 번에서 나오므로 프레임과 주파수 칸이 정확히 맞는다.
#   "spec"  : log 스펙트로그램. 주파수마다 소리의 세기. 0~8000Hz 전체를 31.25Hz 간격으로 고르게 본다.
#             (특정 대역만 골라 보지 않는다. 어느 대역에 흔적이 있을지는 fake를 보고 정하는 것이라 real-only에 어긋난다)
#   "phase" : 위상 변화 편차. 주파수마다 "다음 프레임에서 위상이 예상만큼 돌았는가". 세기만 보는 스펙트로그램이 버리는 정보다.
#             (신경망 보코더는 세기는 잘 흉내 내도 위상을 자연스럽게 잇기 어렵다고 알려져 있다)
#   "residual" : 잔차 스펙트로그램. log 스펙트로그램에서 매끈한 포락선(입 모양 = 성도가 만드는 포먼트)을 뺀 것.
#             성대에서 나온 소리(음원)의 미세 구조만 남는다. 화자마다 다른 입 모양의 영향을 줄이려는 목적
#             (포락선은 LPC 대신 켑스트럼으로 구한다: 같은 256칸 격자에서 바로 계산되고, 하는 일은 같다)
#   "delta" : 시간 변화량. 이웃 프레임과의 log 스펙트로그램 차이. 소리가 얼마나 매끈하게(또는 들쭉날쭉) 변하는지
# 1단계: ["spec"], ["phase"]를 따로 학습 (2026-11-24 완료)
# 2단계(시간 절약): 4개를 쌓아 한 번에 학습하고, 평가에서 채널별 오차를 따로 저장해 채널 조합마다 점수를 따로 본다.
FEATURES = ["spec", "phase", "residual", "delta"]
N_CHANNELS = len(FEATURES)  # U-Net 입력 채널 수
CEPSTRUM_LIFTER = 30  # 잔차용: 켑스트럼 앞쪽 30칸(= 1.9ms보다 느린 변화)만 남겨 포락선을 만든다.
# 포먼트(입 모양)는 이보다 느리게, 목소리 높이(배음 간격, 주기 2.5~15ms)는 이보다 빠르게 변해서 둘이 갈린다.
N_BINS = N_FFT // 2  # 세로 칸 수. 257칸 중 맨 위 한 칸(8000Hz)을 버리고 256칸을 쓴다. (U-Net이 두 번 반으로 줄이므로 4로 나눠떨어져야 한다)


# ===============================
# 5. 조각 길이
# ===============================

SEGMENT_SEC = 2.0  # 모델에 넣는 조각 하나의 길이(초)
MIN_SEGMENT_SEC = 1.0  # 검증·평가에서 이보다 짧게 남은 마지막 조각은 버린다.

SEGMENT_FRAMES = 1 + int(SEGMENT_SEC * SAMPLE_RATE) // HOP_LENGTH  # 2초 = 126프레임
MIN_SEGMENT_FRAMES = 1 + int(MIN_SEGMENT_SEC * SAMPLE_RATE) // HOP_LENGTH  # 1초 = 63프레임


# ===============================
# 6. Diffusion
# ===============================

TIMESTEPS = 1000  # 노이즈를 섞는 전체 단계 수
BETA_START = 0.0001  # 첫 단계에서 추가할 노이즈의 크기
BETA_END = 0.02  # 마지막 단계에서 추가할 노이즈의 크기 (선형 스케줄)


# ===============================
# 7. U-Net
# ===============================

BASE_CHANNELS = 32  # U-Net 첫 단계의 채널 수 (32 → 64 → 128)
TIME_EMBEDDING_DIM = 128  # timestep 숫자 하나를 128개의 특징값으로 바꾼다.
GROUP_NORM_GROUPS = 8  # GroupNorm에서 채널을 몇 묶음으로 나눠 정규화할지 (32, 64, 128 모두 나눠떨어진다)
UNET_FRAMES = 4 * ((SEGMENT_FRAMES + 3) // 4)  # U-Net 안에서 프레임 수를 4의 배수로 채운다. (126 → 128, 두 번 반으로 줄여도 딱 나눠지도록)


# ===============================
# 8. 학습
# ===============================

BATCH_SIZE = 32  # 한 번에 학습할 조각 수
LEARNING_RATE = 0.0001  # 가중치를 한 번에 수정하는 정도 (AdamW)
MAX_EPOCHS = 100  # 최대 학습 횟수
PATIENCE = 10  # 이 횟수만큼 val loss가 좋아지지 않으면 학습을 멈춘다. (early stopping)
GRAD_CLIP = 1.0  # gradient가 갑자기 커지는 것을 막는 상한
SAVE_EVERY = 10  # 이 epoch마다 체크포인트를 따로 저장한다. (나중에 분석용)
NUM_WORKERS = 0  # 데이터를 불러오는 별도 프로세스 수 (Windows에서는 0이 안전하다)

VAL_SEED = 1234  # 검증용 t와 노이즈를 뽑는 전용 난수 시작값 (학습 난수와 섞지 않는다)
VAL_T_BIN_SIZE = 200  # val loss를 t 구간(0–200, 200–400, …, 800–1000)별로 나눠 기록할 때 구간 폭
LOSS_EPS = 1e-8  # masked MSE에서 mask=1 칸이 0개일 때 0으로 나누지 않도록 막는 값


# ===============================
# 9. 점수와 판정
# ===============================

# 점수에 쓸 t. 원래 계획은 100/300/500이었지만, 지난 실험(Mel, XLS-R) 모두 t=500 이상은 real/fake를 거의 못 갈랐고
# 50~200이 가장 좋았다. 그래서 이번 입력의 결과를 보기 전에 50/100/200으로 정해 둔다. (입력이 바뀌면 3단계에서 다시 확인)
SCORE_TIMESTEPS = [50, 100, 200]
# 측정은 더 넓은 t에서 해 두고(segments.csv), 점수 방식·t 조합은 나중에 분석 스크립트에서 다시 계산한다. (평가를 여러 번 돌리지 않으려고)
# 기본 점수(score_mean, score_max)는 위 SCORE_TIMESTEPS만 쓴다.
MEASURE_TIMESTEPS = [10, 25, 50, 100, 200, 400]
NUM_BANDS = 8  # 256칸을 32칸(1kHz)씩 8개 대역으로 나눠 오차를 따로 잰다.
SCORE_NOISE_SEEDS = [1729, 2718]  # t마다 사용할 고정 노이즈의 시작값 (모든 파일이 같은 노이즈로 시험을 본다)
INFER_HOP_SEC = 1.0  # 평가할 때 2초 창을 1초씩 밀면서 자른다.
INFER_HOP_FRAMES = int(INFER_HOP_SEC * SAMPLE_RATE) // HOP_LENGTH  # 1초 = 62프레임

THRESHOLD_PERCENTILE = 95  # dev bona fide 파일 점수의 95백분위를 임계값으로 쓴다.


# ===============================
# 10. RDM: XLS-R 파일 벡터 위의 디퓨전 (rdm_features.py, rdm_train.py, rdm_evaluate.py)
# ===============================

# 음성 파일 하나 → XLS-R 22층 프레임 특징 → 프레임 평균 = 벡터 하나(1024). 그 벡터들 위에서 작은 MLP 디퓨전을 real만으로 학습한다.
# XLS-R은 숫자로 바꾸는 일만 하고(학습하지 않음), 판정은 디퓨전의 노이즈 예측 오차로 한다.
# 프레임마다 디퓨전을 돌리면 오차가 "그 순간 소리가 얼마나 복잡한가"에 흔들렸다(EER 20%). 파일 평균으로 그 흔들림을 없앤다. (EER 11%)
XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"  # 사전학습 음성 모델 (128개 언어, 약 3억 가중치). 처음 실행 때 내려받는다.
XLSR_LAYER = 22  # 24층 중 몇 번째 층을 쓸지. 19~24층 중 22층 (층 후보는 eval을 본 비교에서 나왔다 → 발표 때 밝힌다)
XLSR_DIM = 1024  # XLS-R 한 층의 특징 차원 = 디퓨전 입력 크기
# 파일 벡터는 데이터·증강마다 이 폴더에 한 번만 뽑아 두고, 여러 모델이 나눠 쓴다. (예: 2019_train_clean.npz, ko_train_aug.npz)
RDM_FEATURE_DIR = os.path.join(OUTPUT_DIR, f"rdm_features_xlsr_l{XLSR_LAYER}")

# 모델 두 개 (논문용 = 영어, 시연용 = 영어 + 한국어 + 증강). 아래 세 줄만 바꿔서 고른다.
#   영어:  RDM_RUN_NAME = "rdm_xlsr_l22",       RDM_TRAIN_DATA = ["2019_train"], RDM_DEV_DATA = ["2019_dev"], RDM_AUGMENT = False  (EER 11.49%)
#   시연:  RDM_RUN_NAME = "rdm_xlsr_l22_multi", 아래 값
RDM_RUN_NAME = "rdm_xlsr_l22_multi"  # 결과 폴더 이름 (outputs/<RDM_RUN_NAME>/)
RDM_TRAIN_DATA = ["2019_train", "ko_train", "kspon_train"]  # 학습할 real (영어 + 한국어)
RDM_DEV_DATA = ["2019_dev", "ko_dev", "kspon_dev"]  # 조기 종료·보정·임계값용 real (학습에 안 쓴 화자들, 깨끗한 그대로)
RDM_AUGMENT = True  # True면 학습 real마다 증강본 하나를 더한다. (방 환경 또는 배속 중 무작위 하나)
RDM_DIR = os.path.join(OUTPUT_DIR, RDM_RUN_NAME)

RDM_HIDDEN = 1024  # MLP 안쪽 크기
RDM_BLOCKS = 3  # MLP 잔차 블록 수
RDM_DROPOUT = 0.1  # real 표본이 2,580개뿐이라 외우지 않도록 일부 값을 무작위로 끈다.
RDM_BATCH_SIZE = 256  # 벡터는 작아서 한 번에 많이 넣는다.
RDM_STEPS = 40000  # 학습 걸음 수 (걸음 하나 = 256개 묶음 하나). 데이터 양이 달라도 학습 길이가 같도록 epoch 대신 걸음으로 센다.
RDM_VAL_EVERY = 500  # 이 걸음마다 dev real 손실을 재고, 가장 낮았던 모델을 남긴다.
RDM_NOISE_COUNT = 16  # 점수를 낼 때 t마다 고정 노이즈 16개로 시험을 보고 평균한다. (벡터 하나라 싸다)
# 점수용 t는 SCORE_TIMESTEPS(50/100/200), 임계값은 THRESHOLD_PERCENTILE(95)을 그대로 쓴다.


# ===============================
# 11. 웹 데모 (server.py, explain.py)
# ===============================

WEB_DIR = os.path.join(PROJECT_ROOT, "web")  # index.html이 있는 폴더
MAX_UPLOAD_MB = 20  # 업로드 크기 상한
MAX_AUDIO_SEC = 30.0  # 이보다 긴 음성은 앞부분만 분석한다. (GPU 메모리, 응답 시간)
SERVER_HOST = "127.0.0.1"  # 내 컴퓨터에서만 접속 가능. "0.0.0.0"으로 열면 음성이 네트워크로 오가므로 HTTPS가 필요하다.
SERVER_PORT = 8000  # 브라우저 주소: http://127.0.0.1:8000


# ===============================
# 12. 개발용 (작게 돌려 볼 때)
# ===============================

FILE_LIMIT = None  # 숫자를 넣으면 train·dev에서 앞에서부터 그 개수의 파일만 쓴다. (예: 100)
EVAL_LIMIT = 8000  # 숫자를 넣으면 eval에서 무작위로 그 개수만 평가한다. 지금까지 비교에 쓴 표본과 같다. (None = 전체)


# ===============================
# 13. 실행 장치와 난수
# ===============================

SEED = 42  # 실험을 반복해도 같은 결과가 나오도록 난수 시작값을 고정한다. (eval 표본 고르기에도 쓴다)

if torch.cuda.is_available():  # NVIDIA GPU를 사용할 수 있으면
    DEVICE = torch.device("cuda")  # GPU로 계산한다.
else:
    DEVICE = torch.device("cpu")  # 아니면 CPU로 계산한다.

torch.manual_seed(SEED)  # PyTorch 난수 시작값을 고정한다.
