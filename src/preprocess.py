# ===============================
# preprocess.py
# 음성 파일을 U-Net에 넣을 수 있는 log-Mel로 바꿔서 저장한다.
# FEATURE_TYPE = "wavlm"이면 log-Mel 대신 WavLM 특징 (768, 프레임 수)을 같은 방식으로 저장한다.
# (파일·변수 이름은 처음 만든 대로 mel이라고 두었다. "모델 입력"이라는 뜻으로 읽으면 된다.)
#
# 만드는 파일 (outputs/preprocess 폴더)
#   mel_stats.pt         : train real로 계산한 Mel 칸별 평균·표준편차
#   train_full_mels.pt   : train real 파일마다 전체 길이의 log-Mel (학습 때 무작위로 2초씩 자른다)
#   val_mels.pt          : dev real을 고정 2초 조각으로 자른 log-Mel (검증·임계값용)
#   mel_example.png      : 확인용 Mel 그림
#   codec_examples/      : 같은 문장을 원본과 학습 코덱별로 저장한 wav (AUGMENT_CODEC = True일 때, 직접 들어 보기용)
#
# 이 파일의 함수(load_audio, wave_to_logmel 등)는 evaluate.py에서도 가져다 쓴다.
# ===============================

import csv  # ESC-50의 파일 목록(esc50.csv)을 읽는다. (환경음 증강)
import os  # 파일 경로를 다룬다.
import subprocess  # ffmpeg 프로그램을 실행한다. (코덱 증강)

import imageio_ffmpeg  # venv 안에 설치된 ffmpeg 실행 파일의 위치를 알려 준다.
import librosa  # 음성 파일을 읽고 Mel-spectrogram을 계산한다.
import matplotlib  # 그림을 저장한다.
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt  # 그림을 그린다.
import numpy as np  # 숫자 배열을 계산한다.
import soundfile as sf  # 코덱 증강 예시를 wav 파일로 저장한다. (직접 들어 보기용)
import torch  # 결과를 PyTorch 텐서로 저장한다.
from scipy.signal import butter, fftconvolve, sosfilt  # pstn 흉내: 일부 대역만 통과시키는 필터 / 방 울림: 말소리에 잔향 응답을 입힌다.
from transformers import AutoFeatureExtractor, AutoModel  # 미리 학습된 음성 모델(WavLM, wav2vec2)을 이름으로 불러온다. (FEATURE_TYPE = "wavlm"일 때만 쓴다)

from config import *  # config.py의 모든 설정값을 가져온다.


# ===============================
# 1. 라벨(프로토콜) 읽기
# ===============================

def read_protocol(data_name):  # data_name: config.py의 DATASETS 이름 (예: "2019_train", "2021_eval")

    dataset = DATASETS[data_name]  # 라벨 파일, 음성 폴더, 칸 번호

    if "protocol" not in dataset:  # 라벨 파일이 없는 데이터(한국어)는 폴더를 뒤져서 목록을 만든다.
        return read_folder(data_name)

    columns = dataset["columns"]  # 예: {"file_id": 1, "label": 4, ...}
    min_parts = max(columns.values()) + 1  # 필요한 칸이 모두 있으려면 최소 이만큼 조각이 있어야 한다.

    items = []  # 파일 정보를 하나씩 담을 목록

    with open(dataset["protocol"], "r", encoding="utf-8") as f:  # 라벨 파일을 연다.
        for line in f:  # 한 줄씩 읽는다.
            parts = line.strip().split()  # 예: "LA_0079 LA_T_1138215 - - bonafide" → 5조각
            if len(parts) < min_parts:  # 칸이 모자란 줄(빈 줄 등)은 건너뛴다.
                continue

            items.append({  # 파일 하나의 정보를 저장한다.
                "path": os.path.join(dataset["audio_dir"], parts[columns["file_id"]] + ".flac"),  # 음성 파일 경로
                "speaker_id": parts[columns["speaker_id"]],  # 화자 ID
                "file_id": parts[columns["file_id"]],  # 파일 ID
                "attack_id": parts[columns["attack_id"]],  # 공격 종류 (2019 real은 "-")
                "label": parts[columns["label"]],  # "bonafide"(real) 또는 "spoof"(fake)
            })

    if len(items) == 0:  # 한 줄도 못 읽었으면 칸 번호나 라벨 파일이 잘못된 것이다. (조용히 0개로 넘어가지 않도록)
        raise ValueError(f"{data_name}: 라벨을 한 줄도 읽지 못했다. config.py의 protocol 경로와 columns를 확인할 것")

    return items  # 파일 정보 목록을 돌려준다.


def read_folder(data_name):  # 라벨 파일 없이 폴더 안의 wav를 모두 real로 읽는다. (한국어 데이터용)

    dataset = DATASETS[data_name]  # 음성 폴더, 이 분할에 넣을 화자 목록
    items = []  # read_protocol과 같은 모양의 파일 정보 목록

    for folder, sub_folders, files in os.walk(dataset["audio_dir"]):  # 음성 폴더 아래의 모든 폴더를 하나씩 본다.
        sub_folders.sort()  # 폴더를 이름순으로 돌아서, 언제 실행해도 파일 순서가 같게 한다.
        speaker_id = os.path.basename(folder)  # 화자마다 폴더 하나라서 폴더 이름을 화자 ID로 쓴다.

        if not any(key in speaker_id for key in dataset["speakers"]):  # 이 분할의 화자가 아니면
            continue  # 건너뛴다. (train 화자가 dev에 섞이지 않게)

        for name in sorted(files):  # 파일 이름순으로
            if not name.lower().endswith(".wav"):  # wav가 아닌 파일은 건너뛴다.
                continue
            items.append({
                "path": os.path.join(folder, name),  # 음성 파일 경로
                "speaker_id": speaker_id,  # 화자 ID (폴더 이름)
                "file_id": os.path.splitext(name)[0],  # 파일 ID (확장자를 뺀 이름. 화자 이름이 들어 있어 겹치지 않는다)
                "attack_id": "-",  # 공격 없음
                "label": "bonafide",  # 모두 진짜 사람 음성
            })

    if len(items) == 0:  # 하나도 못 찾았으면 폴더 경로나 화자 이름이 잘못된 것이다.
        raise ValueError(f"{data_name}: wav를 하나도 찾지 못했다. config.py의 audio_dir와 speakers를 확인할 것")

    speakers = sorted({item["speaker_id"] for item in items})  # 실제로 들어간 화자들
    print(f"[{data_name}] 화자 {len(speakers)}명, 파일 {len(items)}개: {speakers}")  # 화자가 의도대로 나뉘었는지 확인한다.

    return items


# ===============================
# 2. 음성 → log-Mel
# ===============================

def load_audio(path):  # 음성 파일 하나를 읽는다.

    wave, _ = librosa.load(  # 파일을 읽어 숫자 배열(waveform)로 바꾼다.
        path,
        sr=SAMPLE_RATE,  # 16kHz로 맞춘다. (이미 16kHz면 그대로)
        mono=True,  # 모노로 맞춘다.
    )
    # 파일별 Mel 정규화 같은 "음색을 바꾸는" 정규화는 하지 않는다. 합성 흔적(아티팩트)을 지울 수 있어서.

    if TRIM_SILENCE:  # 무음 자르기 실험일 때만
        wave, _ = trim_silence(wave)

    if NORMALIZE_PEAK:  # 소리 크기만 맞춘다. (ASVspoof는 이미 1.0이라 그대로)
        wave = normalize_peak(wave)

    return wave.astype(np.float32)  # 32비트 실수로 돌려준다.


def normalize_peak(wave):  # 가장 큰 진폭이 1.0이 되도록 전체에 같은 숫자를 곱한다. 웹 분석도 이 함수를 쓴다.

    peak = float(np.abs(wave).max())
    return wave / peak if peak > 0 else wave  # 완전히 0인 소리는 그대로 둔다.


def trim_silence(wave):  # 앞뒤 무음을 잘라 낸다. 반환: (잘린 소리, 앞에서 잘라 낸 길이(샘플)). 웹 분석도 이 함수를 쓴다.

    trimmed, (start, _) = librosa.effects.trim(  # 앞뒤에서 가장 큰 소리보다 40dB 이상 작은 부분을 잘라 낸다. (가운데 쉼은 그대로)
        wave,
        top_db=-SILENCE_DB_THRESHOLD,  # 40: 무음 mask와 같은 기준
        frame_length=WIN_LENGTH,  # 25ms 창으로 소리 크기를 잰다.
        hop_length=WIN_LENGTH // 2,  # 12.5ms씩 옮긴다.
    )
    return trimmed, int(start)  # start: 웹에서 원래 녹음과 분석 결과의 시간을 맞출 때 쓴다.


def wave_to_logmel(wave):  # waveform → log-Mel 또는 log 스펙트로그램 (N_BINS, 프레임 수)

    if FEATURE_TYPE == "mel":  # Mel을 쓰는 경우 (원래 방식)
        mel = librosa.feature.melspectrogram(  # STFT → 파워 → Mel 필터를 한 번에 계산한다.
            y=wave,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,  # 크기의 제곱(파워)을 사용한다.
        )
    else:  # Mel 필터를 거치지 않는 경우: STFT 결과를 그대로 쓴다.
        stft = librosa.stft(  # 프레임마다 주파수 성분을 계산한다. (257, 프레임 수): 0Hz ~ 8000Hz를 31.25Hz 간격으로
            y=wave,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,  # Mel과 같은 설정이라 프레임 수도 같다.
        )
        mel = np.abs(stft) ** 2  # 크기의 제곱(파워). Mel 방식의 power=2.0과 같다. (변수 이름은 아래 코드와 맞추려고 mel로 둔다)
        mel = mel[:N_BINS]  # 맨 위 한 칸(8000Hz)을 버려 256칸으로 맞춘다.
    # Mel은 이 파워 스펙트로그램의 여러 칸을 묶어 80칸으로 줄인 것이다. 묶지 않으면 모든 주파수를 같은 간격으로 본다.

    logmel = np.log(mel + LOG_EPS)  # 값의 차이가 너무 크지 않도록 log로 압축한다.

    return logmel.astype(np.float32)  # (N_BINS, 프레임 수)


WAVLM = {}  # 불러온 WavLM을 담아 둘 곳. 파일마다 다시 불러오지 않도록 처음 한 번만 채운다.


def wave_to_wavlm(wave):  # waveform → 음성 모델 특징 (WAVLM_DIM, 프레임 수). WavLM, wav2vec2 모두 이 함수를 쓴다.

    if "model" not in WAVLM:  # 처음 부를 때만
        WAVLM["model"] = AutoModel.from_pretrained(WAVLM_MODEL_NAME).to(DEVICE).eval()  # 이름에 맞는 모델(WavLM, wav2vec2)을 불러와 GPU에 올린다. (학습하지 않고 그대로 쓴다)
        WAVLM["normalize"] = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_NAME).do_normalize  # 사전학습 때 입력을 정규화했는지
        print(f"음성 모델 불러옴: {WAVLM_MODEL_NAME}, {WAVLM_LAYER}층 사용, 입력 정규화 {WAVLM['normalize']}")

    if WAVLM["normalize"]:  # XLS-R 등: 사전학습 때처럼 파일마다 평균 0, 표준편차 1로 맞춰 넣는다. (Hugging Face 전처리와 같은 식)
        wave = (wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)
    x = torch.from_numpy(wave.astype(np.float32))[None].to(DEVICE)  # (1, 샘플 수). wavlm-base-plus는 waveform을 그대로 받는다.
    with torch.no_grad():  # 음성 모델은 학습하지 않는다.
        out = WAVLM["model"](x, output_hidden_states=True)  # 모든 층의 출력을 받는다.
    hidden = out.hidden_states[WAVLM_LAYER][0]  # (프레임 수, WAVLM_DIM): 고른 층의 출력. 20ms마다 숫자 WAVLM_DIM개

    return hidden.T.cpu().numpy().astype(np.float32)  # (WAVLM_DIM, 프레임 수): Mel과 같은 (세로 칸, 프레임) 모양으로 맞춘다.


def make_silence_mask(wave, num_frames):  # 무음 프레임은 0, 소리 있는 프레임은 1

    if FEATURE_TYPE == "wavlm":  # WavLM 프레임 i는 샘플 [320i, 320i + 400) 구간을 본다.
        rms = librosa.feature.rms(y=wave, frame_length=WIN_LENGTH, hop_length=WAVLM_HOP, center=False)[0]  # 같은 구간으로 소리 크기를 재야 프레임이 맞는다. (WIN_LENGTH = 400샘플)
    else:
        rms = librosa.feature.rms(  # 프레임마다 소리 크기(RMS 에너지)를 계산한다.
            y=wave,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,  # Mel과 같은 간격으로 계산해야 프레임이 맞는다.
        )[0]

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)  # 가장 큰 프레임을 0dB로 놓고 dB로 바꾼다.
    silence_mask = (rms_db > SILENCE_DB_THRESHOLD).astype(np.float32)  # 기준보다 크면 1(소리 있음)

    return silence_mask[:num_frames]  # Mel 프레임 수에 맞춰 자른다.


def make_frame_mask(wave, num_frames):  # 프레임마다 학습·점수에 쓸지(1) 뺄지(0)

    mask = np.ones(num_frames, dtype=np.float32)  # 처음에는 모든 프레임을 사용한다.

    if MASK_SILENCE:  # 무음 mask를 켠 경우에만
        mask = mask * make_silence_mask(wave, num_frames)  # 무음 프레임을 0으로 바꾼다.

    return mask  # (프레임 수,)


def normalize_logmel(logmel, stats):  # train 통계로 정규화한다. (평균 0, 표준편차 1 근처)

    mean = stats["mean"][:, None]  # (80, 1): Mel 칸마다 평균
    std = stats["std"][:, None]  # (80, 1): Mel 칸마다 표준편차

    return (logmel - mean) / std  # 모든 파일에 같은 통계를 쓴다. (파일별 정규화는 하지 않는다)


def wave_to_input(wave, stats=None):  # waveform → 모델 입력과 mask (코덱 증강한 소리도 이 함수로 똑같이 처리한다)

    if FEATURE_TYPE == "wavlm":  # WavLM 특징을 쓰는 경우
        logmel = wave_to_wavlm(wave)  # (768, 프레임 수). 아래 mask·정규화는 Mel과 똑같이 한다.
    else:
        logmel = wave_to_logmel(wave)  # log-Mel로 바꾼다.
    mask = make_frame_mask(wave, logmel.shape[1])  # 프레임별 mask를 만든다. (코덱을 거친 소리면 그 소리 기준으로. 평가 때와 같은 조건)

    if stats is not None:  # 통계가 주어지면
        logmel = normalize_logmel(logmel, stats)  # 정규화한다.

    return logmel, mask  # (80, 프레임 수), (프레임 수,)


def file_to_logmel(path, stats=None):  # 파일 하나 → (정규화된) log-Mel과 mask

    return wave_to_input(load_audio(path), stats)  # 음성을 읽어서 모델 입력으로 바꾼다.


# ===============================
# 2-1. 코덱 증강 (학습 데이터에만 쓴다)
# ===============================

# 진짜 음성을 실제 전화 코덱에 통과시킨다. ffmpeg로 인코딩(압축)했다가 다시 디코딩(복원)하면, 코덱이 지운 정보는 돌아오지 않는다.
#   예: alaw는 8kHz로 보내므로 4kHz 위의 높은 소리가 사라지고, 8비트로 압축하면서 작은 잡음이 생긴다.
# 마지막에는 16kHz로 되돌려 모델 입력 조건(샘플링, 길이)을 원래와 똑같이 맞춘다.

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # venv 안의 ffmpeg 실행 파일 경로


def run_ffmpeg(args, data):  # ffmpeg에 소리 데이터(bytes)를 넣고 결과(bytes)를 받는다. 파일을 따로 만들지 않는다.

    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", *args],  # 진행 메시지는 숨기고 오류만 보인다.
        input=data,  # pipe:0 = 이 데이터를 입력으로
        stdout=subprocess.PIPE,  # pipe:1 = 결과를 돌려받는다.
        stderr=subprocess.PIPE,
        check=True,  # ffmpeg가 실패하면 바로 오류를 낸다. (조용히 빈 소리로 넘어가지 않도록)
    )

    return result.stdout


def apply_codec(wave, codec):  # codec: CODEC_SETTINGS의 이름 ("alaw", "mulaw", "g722")

    codec_format, codec_rate = CODEC_SETTINGS[codec]  # ffmpeg 형식 이름, 코덱의 샘플링 주파수
    pcm = np.clip(wave, -1.0, 1.0).astype(np.float32).tobytes()  # 16kHz 32비트 실수 소리를 bytes로

    encoded = run_ffmpeg(  # 인코딩: 16kHz 소리 → 코덱 샘플링으로 바꾸고 → 코덱으로 압축
        ["-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0", "-ar", str(codec_rate), "-f", codec_format, "pipe:1"],
        pcm,
    )
    input_format = ["-f", codec_format]  # 압축된 데이터가 어떤 코덱인지
    if codec_format != "g722":  # alaw·mulaw 데이터에는 정보 머리말이 없어서 샘플링 주파수와 채널 수를 직접 알려 줘야 한다. (g722는 알아서 16kHz로 읽는다)
        input_format += ["-ar", str(codec_rate), "-ac", "1"]
    decoded = run_ffmpeg(  # 디코딩: 압축된 소리 → 풀고 → 다시 16kHz 32비트 실수로
        [*input_format, "-i", "pipe:0", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        encoded,
    )

    back = np.frombuffer(decoded, dtype=np.float32).copy()  # bytes → 숫자 배열 (copy: 읽기 전용 배열을 고칠 수 있는 배열로)
    return librosa.util.fix_length(back, size=len(wave))  # 길이를 원래와 똑같이 맞춘다. (코덱 지연·리샘플링으로 몇 샘플 달라질 수 있다)


def apply_augment(wave, augment, rng):  # augment: AUGMENT_CODECS 중 하나. rng: pstn 대역 경계를 고를 난수

    if augment != "pstn":  # alaw, g722: 코덱에 그대로 통과시킨다.
        return apply_codec(wave, augment)

    # pstn 흉내: 일반 전화망이 통과시키는 대역만 남기고 → μ-law로 녹음한다. (2021 pstn 조건의 마지막 단계가 μ-law)
    low_hz = rng.uniform(*PSTN_LOW_HZ_RANGE)  # 낮은 쪽 경계를 파일마다 조금씩 다르게
    high_hz = rng.uniform(*PSTN_HIGH_HZ_RANGE)  # 높은 쪽 경계도
    bandpass = butter(4, (low_hz, high_hz), btype="bandpass", fs=SAMPLE_RATE, output="sos")  # 그 대역만 통과시키는 필터
    wave = sosfilt(bandpass, wave).astype(np.float32)  # 대역 밖의 소리를 줄인다.

    return apply_codec(wave, "mulaw")


def save_codec_examples(path, out_dir):  # 같은 파일을 원본과 학습 코덱별로 저장한다. (직접 들어 보고 확인하기용)

    os.makedirs(out_dir, exist_ok=True)
    wave = load_audio(path)
    rng = np.random.default_rng(SEED)  # pstn 예시의 대역 경계용
    sf.write(os.path.join(out_dir, "original.wav"), wave, SAMPLE_RATE)
    for augment in AUGMENT_CODECS:
        sf.write(os.path.join(out_dir, f"{augment}.wav"), apply_augment(wave, augment, rng), SAMPLE_RATE)


# ===============================
# 2-2. 환경음 증강 (학습: NOISE_TRAIN_FOLDS, 잡음 평가: NOISE_EVAL_FOLDS)
# ===============================

NOISE_BANK = {}  # 묶음 번호들 → 환경음 목록. 처음 한 번만 읽는다.


def active_rms(wave):  # 소리가 있는 부분만의 크기 (ESC-50은 소리가 짧으면 뒤를 디지털 무음으로 채워 둔 파일이 많다)

    rms = librosa.feature.rms(y=wave, frame_length=WIN_LENGTH, hop_length=WIN_LENGTH // 2, center=False)[0]  # 25ms 창마다 크기
    if rms.max() == 0:  # 완전히 0인 소리
        return 0.0
    active = rms[rms > rms.max() * 10 ** (SILENCE_DB_THRESHOLD / 20)]  # 가장 큰 곳보다 40dB 이상 작은 부분(무음)은 뺀다.
    # 무음까지 넣어 크기를 재면, 무음이 긴 잡음일수록 실제 소리보다 작게 재져서 섞을 때 너무 크게 키우게 된다.

    return float(np.sqrt(np.mean(active ** 2)))  # 소리 있는 부분의 평균 크기 (RMS)


def load_noise_bank(folds):  # ESC-50에서 이 묶음들의 환경음을 읽는다. (목소리 성분이 강한 NOISE_EXCLUDE 6종은 뺀다)

    key = tuple(folds)
    if key not in NOISE_BANK:  # 처음 부를 때만 파일을 읽는다.
        clips = []
        with open(os.path.join(NOISE_ROOT, "meta", "esc50.csv"), "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):  # 한 줄 = 환경음 파일 하나 (filename, fold, category, …)
                if int(row["fold"]) not in folds or row["category"] in NOISE_EXCLUDE:
                    continue
                clip, _ = librosa.load(os.path.join(NOISE_ROOT, "audio", row["filename"]), sr=SAMPLE_RATE, mono=True)  # 44.1kHz → 16kHz
                if active_rms(clip) > 0:  # 완전히 조용한 파일은 잡음으로 쓸 수 없다.
                    clips.append(clip.astype(np.float32))
        NOISE_BANK[key] = clips
        print(f"환경음 묶음 {list(folds)}: {len(clips)}개 (제외한 종류 {len(NOISE_EXCLUDE)}개)")

    return NOISE_BANK[key]


def pad_for_noise(wave, rng):  # 말소리 앞뒤에 무작위 길이의 빈 구간을 붙인다. (환경음·방 환경 증강이 같이 쓴다)

    pad_front = int(rng.uniform(*NOISE_PAD_SEC_RANGE) * SAMPLE_RATE)  # 앞 빈 구간 길이 (샘플)
    pad_back = int(rng.uniform(*NOISE_PAD_SEC_RANGE) * SAMPLE_RATE)  # 뒤 빈 구간 길이 (앞과 따로 고른다)
    return np.concatenate([np.zeros(pad_front, np.float32), wave, np.zeros(pad_back, np.float32)])
    # 빈 구간은 부른 쪽에서 반드시 잡음으로 덮는다. 잡음 없이 빈 구간만 붙이면 무음 길이가 다시 단서가 되므로 절대 따로 쓰지 않는다.


def add_noise(wave, clips, snr_db, rng):  # 말소리 앞뒤에 무작위 길이의 빈 구간을 붙이고, 전체에 환경음을 SNR에 맞춰 섞는다.

    padded = pad_for_noise(wave, rng)  # 앞뒤 빈 구간 (아래에서 잡음으로 덮인다)

    clip = clips[rng.integers(len(clips))]  # 환경음 하나를 무작위로
    start = rng.integers(len(clip))  # 그 안에서 시작 위치도 무작위로
    noise = np.resize(np.roll(clip, -start), len(padded))  # 시작 위치부터 읽고, 길이가 모자라면 처음으로 돌아가 이어 붙인다.

    speech_rms = active_rms(wave)  # 말소리 크기 (앞뒤 빈 구간을 붙이기 전, 말소리만으로 잰다)
    noise_rms = active_rms(clip)  # 잡음 크기는 잘라 낸 조각이 아니라 환경음 파일 전체에서 잰다. (소리 있는 부분만)
    # 짧은 조각이 하필 그 파일의 무음 부분에 걸리면 조각의 크기가 0이 되어 나눌 수 없다. 파일 전체로 재면 항상 0보다 크다.
    # 그래서 "띄엄띄엄 나는 소리"(개 짖는 소리 등)는 말소리 구간과 안 겹칠 수도 있다. 실제 녹음에서도 그렇다.
    noise = noise * (speech_rms / noise_rms) / 10 ** (snr_db / 20)  # SNR(dB) = 20·log10(말소리 크기 / 잡음 크기)가 되도록 잡음 크기를 맞춘다.
    # 말소리 크기에 맞춰 잡음을 키우거나 줄이므로, real·fake의 원래 음량이 달라도 같은 SNR이면 같은 비율로 섞인다.

    return (padded + noise).astype(np.float32)  # 진폭 정규화는 하지 않는다. (원래 방식과 같게)


def save_noise_examples(path, out_dir):  # 같은 파일에 SNR별로 잡음을 섞어 저장한다. (직접 들어 보고 확인하기용)

    os.makedirs(out_dir, exist_ok=True)
    wave = load_audio(path)  # TRIM_SILENCE면 앞뒤 무음이 잘린 말소리
    clips = load_noise_bank(NOISE_TRAIN_FOLDS)
    sf.write(os.path.join(out_dir, "original.wav"), wave, SAMPLE_RATE)
    for snr_db in [20, 10, 5]:
        noisy = add_noise(wave, clips, snr_db, np.random.default_rng(SEED))  # 같은 seed → 같은 잡음·빈 구간, SNR만 다르다.
        sf.write(os.path.join(out_dir, f"snr{snr_db}.wav"), noisy, SAMPLE_RATE)


# ===============================
# 2-3. 방 환경 증강 (방 울림 + 핑크 노이즈, 코드로 직접 만든다)
# ===============================

def pink_noise(length, rng):  # 핑크 노이즈: 저음이 크고 고음으로 갈수록 약해지는 잡음 (방·마이크의 바닥 잡음과 비슷)

    spectrum = np.fft.rfft(rng.standard_normal(length))  # 화이트 노이즈(모든 주파수가 같은 세기)를 주파수로 바꾼다.
    freqs = np.fft.rfftfreq(length, d=1 / SAMPLE_RATE)  # 각 칸의 주파수(Hz)
    keep = freqs >= ROOM_NOISE_LOW_HZ  # 귀에 거의 안 들리는 초저음은 뺀다.
    spectrum[keep] /= np.sqrt(freqs[keep])  # 세기(power)가 1/주파수가 되도록 → 진폭은 1/√주파수. 주파수가 2배면 세기는 절반(-3dB)
    spectrum[~keep] = 0
    noise = np.fft.irfft(spectrum, length)  # 다시 소리로

    return (noise / np.sqrt(np.mean(noise ** 2))).astype(np.float32)  # 크기(RMS)를 1로 맞춘다. 실제 크기는 add_room에서 SNR로 정한다.


def room_impulse(rt60, drr_db, rng):  # 방의 잔향 응답: "손뼉을 한 번 쳤을 때 마이크에 들리는 소리"

    t = np.arange(1, int(rt60 * SAMPLE_RATE)) / SAMPLE_RATE  # 울림이 이어지는 시간 (직접 들어온 소리 바로 다음부터 RT60까지)
    tail = rng.standard_normal(len(t)) * np.exp(-np.log(1000) * t / rt60)  # 무작위 반사음이 점점 작아진다. RT60초 뒤에 진폭 1/1000 = -60dB
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail ** 2))  # 울림 전체의 에너지를 "직접 들어온 소리(1)보다 DRR dB 작게" 맞춘다.

    return np.concatenate([[1.0], tail])  # 맨 앞 1 = 입에서 마이크로 바로 들어온 소리, 그 뒤 = 벽에 반사된 울림


def add_room(wave, rt60, drr_db, snr_db, rng):  # 방 울림 → 앞뒤 빈 구간 → 핑크 노이즈로 전체 덮기

    wet = fftconvolve(wave, room_impulse(rt60, drr_db, rng)).astype(np.float32)  # 말소리에 방 울림을 입힌다. (끝에 RT60만큼 울림 꼬리가 붙는다)
    padded = pad_for_noise(wet, rng)  # 앞뒤 빈 구간 (아래에서 핑크 노이즈로 덮인다)
    noise = pink_noise(len(padded), rng) * active_rms(wet) / 10 ** (snr_db / 20)  # 말소리 크기에 맞춰 SNR dB만큼 작은 바닥 잡음

    return (padded + noise).astype(np.float32)  # 진폭 정규화는 하지 않는다. (원래 방식과 같게)


def random_room(wave, rng):  # 방 환경 값들을 범위 안에서 무작위로 골라 입힌다. (학습·평가가 같이 쓴다)

    rt60 = rng.uniform(*ROOM_RT60_RANGE)
    drr_db = rng.uniform(*ROOM_DRR_RANGE)
    snr_db = rng.uniform(*ROOM_SNR_RANGE)
    return add_room(wave, rt60, drr_db, snr_db, rng)


def save_room_examples(path, out_dir):  # 같은 파일에 방 환경을 몇 가지로 입혀 저장한다. (직접 들어 보고 확인하기용)

    os.makedirs(out_dir, exist_ok=True)
    wave = load_audio(path)  # TRIM_SILENCE면 앞뒤 무음이 잘린 말소리
    sf.write(os.path.join(out_dir, "original.wav"), wave, SAMPLE_RATE)
    for rt60, drr_db, snr_db in [(ROOM_RT60_RANGE[0], ROOM_DRR_RANGE[1], ROOM_SNR_RANGE[1]),  # 가장 약하게: 작은 방, 마이크 가까이, 조용함
                                 (ROOM_RT60_RANGE[1], ROOM_DRR_RANGE[0], ROOM_SNR_RANGE[0])]:  # 가장 세게: 넓은 방, 마이크 멀리, 잡음 큼
        room = add_room(wave, rt60, drr_db, snr_db, np.random.default_rng(SEED))
        sf.write(os.path.join(out_dir, f"rt{rt60}_drr{drr_db:.0f}_snr{snr_db:.0f}.wav"), room, SAMPLE_RATE, subtype="FLOAT")
        # 울림이 겹치면 최대 진폭이 1을 조금 넘을 수 있다. 기본(16비트)으로 저장하면 그 부분이 잘려 찌그러지므로 실수 그대로 저장한다. (학습에는 영향 없음)


# ===============================
# 3. 2초 조각으로 자르기
# ===============================

def pad_to_segment(logmel_piece, mask_piece):  # SEGMENT_FRAMES보다 짧은 조각의 오른쪽을 채운다.

    missing = SEGMENT_FRAMES - logmel_piece.shape[1]  # 부족한 프레임 수

    if missing > 0:  # 부족하면
        logmel_piece = np.pad(logmel_piece, ((0, 0), (0, missing)), constant_values=0.0)  # 정규화된 값 0(평균)으로 채운다.
        mask_piece = np.pad(mask_piece, (0, missing), constant_values=0.0)  # 채운 부분은 mask=0

    return logmel_piece, mask_piece  # (80, 126), (126,)


def cut_segments(logmel, mask, hop_frames):  # 정규화된 log-Mel을 2초 조각들로 자른다.

    segments = []  # (조각 Mel, 조각 mask, 시작 프레임) 목록
    total_frames = logmel.shape[1]  # 전체 프레임 수

    for start in range(0, total_frames, hop_frames):  # hop_frames 간격으로 시작 위치를 옮긴다.
        end = start + SEGMENT_FRAMES  # 조각의 끝 위치
        piece_len = min(end, total_frames) - start  # 실제로 남은 길이

        if piece_len < MIN_SEGMENT_FRAMES:  # 1초보다 짧게 남았으면
            break  # 버리고 끝낸다.

        piece_mel, piece_mask = pad_to_segment(logmel[:, start:end], mask[start:end])  # 부족하면 채운다.
        segments.append((piece_mel, piece_mask, start))  # 목록에 추가한다.

        if end >= total_frames:  # 파일 끝까지 다 잘랐으면
            break  # 끝낸다.

    return segments


# ===============================
# 4. 정규화 통계 계산 (train real만 사용)
# ===============================

def compute_stats(raw_logmels, masks):  # Mel 칸마다 평균과 표준편차를 구한다.

    total_sum = np.zeros(N_BINS, dtype=np.float64)  # 칸별 합
    total_sq_sum = np.zeros(N_BINS, dtype=np.float64)  # 칸별 제곱합
    total_count = 0  # 사용한 프레임 수

    for logmel, mask in zip(raw_logmels, masks):  # 파일마다
        used = logmel[:, mask > 0]  # mask=1인 프레임만 사용한다.
        total_sum += used.sum(axis=1)
        total_sq_sum += (used.astype(np.float64) ** 2).sum(axis=1)
        total_count += used.shape[1]

    mean = total_sum / total_count  # 평균
    std = np.sqrt(total_sq_sum / total_count - mean ** 2)  # 표준편차 = √(제곱의 평균 − 평균의 제곱)
    std = np.maximum(std, 1e-5)  # 0으로 나누지 않도록 아주 작은 값은 막는다.

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


# ===============================
# 5. 분할별 처리
# ===============================

def select_bonafide(data_name):  # 해당 데이터에서 real(bonafide) 파일만 고른다.

    items = [item for item in read_protocol(data_name) if item["label"] == "bonafide"]  # real만

    if FILE_LIMIT is not None:  # 개발용으로 개수를 줄인 경우
        items = items[:FILE_LIMIT]  # 앞에서부터 FILE_LIMIT개만 사용한다.

    return items


def build_train(stats_out_path, train_out_path):  # train real → 통계 + 파일별 전체 log-Mel

    items = select_bonafide(TRAIN_DATA)  # train real 파일 목록
    print(f"[train] real 파일 {len(items)}개 처리 시작")

    raw_logmels, masks, file_ids = [], [], []  # 정규화 전 Mel, mask, 파일 ID
    rng = np.random.default_rng(SEED)  # 파일마다 어떤 코덱을 입힐지(와 pstn 대역 경계를) 고르는 난수. 언제 돌려도 같은 선택
    noise_clips = load_noise_bank(NOISE_TRAIN_FOLDS) if AUGMENT_NOISE else None  # 학습용 환경음 (평가용 묶음은 쓰지 않는다)

    for i, item in enumerate(items):  # 파일마다
        wave = load_audio(item["path"])  # 음성을 한 번만 읽어서 원본과 증강본에 같이 쓴다.
        logmel, mask = wave_to_input(wave)  # 정규화 전 log-Mel과 mask
        raw_logmels.append(logmel)
        masks.append(mask)
        file_ids.append(item["file_id"])

        if AUGMENT_CODEC:  # 코덱 증강: 같은 파일을 코덱에 통과시킨 소리로 하나 더 만든다.
            augment = AUGMENT_CODECS[rng.integers(len(AUGMENT_CODECS))]  # 학습 코덱 중 하나를 무작위로
            logmel, mask = wave_to_input(apply_augment(wave, augment, rng))
            raw_logmels.append(logmel)
            masks.append(mask)
            file_ids.append(f"{item['file_id']}@{augment}")  # 예: LA_T_1138215@alaw (어떤 코덱을 입혔는지 남긴다)

        if AUGMENT_NOISE:  # 환경음 증강: 같은 파일에 앞뒤 빈 구간 + 환경음을 섞은 소리로 하나 더 만든다.
            snr_db = rng.uniform(*NOISE_SNR_RANGE)  # 파일마다 SNR을 무작위로
            logmel, mask = wave_to_input(add_noise(wave, noise_clips, snr_db, rng))
            raw_logmels.append(logmel)
            masks.append(mask)
            file_ids.append(f"{item['file_id']}@noise{snr_db:.0f}")  # 예: LA_T_1138215@noise12 (몇 dB로 섞었는지 남긴다)

        if AUGMENT_ROOM:  # 방 환경 증강: 같은 파일에 방 울림 + 앞뒤 빈 구간 + 핑크 노이즈를 입힌 소리로 하나 더 만든다.
            logmel, mask = wave_to_input(random_room(wave, rng))
            raw_logmels.append(logmel)
            masks.append(mask)
            file_ids.append(f"{item['file_id']}@room")  # 예: LA_T_1138215@room

        if (i + 1) % 500 == 0:  # 500개마다 진행 상황을 출력한다.
            print(f"  {i + 1}/{len(items)}")

    if AUGMENT_CODEC:  # 코덱이 고르게 섞였는지 확인한다.
        counts = {augment: sum(f.endswith("@" + augment) for f in file_ids) for augment in AUGMENT_CODECS}
        print(f"[train] 코덱 증강: 원본 {len(items)}개 + 증강 {sum(counts.values())}개 {counts}")

    stats = compute_stats(raw_logmels, masks)  # train real로만 통계를 계산한다.
    torch.save({"mean": torch.from_numpy(stats["mean"]), "std": torch.from_numpy(stats["std"])}, stats_out_path)

    mels = [torch.from_numpy(normalize_logmel(m, stats)) for m in raw_logmels]  # 정규화한 뒤 텐서로 바꾼다.
    torch.save({  # 파일마다 길이가 달라서 목록(list)으로 저장한다.
        "mels": mels,  # 각각 (80, 프레임 수)
        "masks": [torch.from_numpy(m) for m in masks],  # 각각 (프레임 수,)
        "file_ids": file_ids,
    }, train_out_path)

    total_frames = sum(m.shape[1] for m in mels)  # 전체 프레임 수
    print(f"[train] 저장 완료: 파일 {len(mels)}개, 전체 {total_frames}프레임 (약 {total_frames * FRAME_HOP / SAMPLE_RATE / 60:.1f}분)")

    return stats, mels[0], torch.from_numpy(masks[0])  # 확인용 그림에 쓸 첫 파일도 돌려준다.


def build_val(stats, val_out_path):  # dev real → 고정 2초 조각

    items = select_bonafide(DEV_DATA)  # dev real 파일 목록
    print(f"[dev] real 파일 {len(items)}개 처리 시작")

    seg_mels, seg_masks, seg_file_ids = [], [], []  # 조각 Mel, 조각 mask, 조각의 원본 파일 ID
    skipped = 0  # 1초보다 짧아서 조각이 하나도 안 나온 파일 수

    for i, item in enumerate(items):  # 파일마다
        logmel, mask = file_to_logmel(item["path"], stats)  # 정규화된 log-Mel과 mask
        segments = cut_segments(logmel, mask, hop_frames=SEGMENT_FRAMES)  # 겹치지 않게 2초씩 자른다.

        if len(segments) == 0:  # 조각이 하나도 없으면
            skipped += 1
            continue

        for piece_mel, piece_mask, _ in segments:  # 조각마다
            seg_mels.append(piece_mel)
            seg_masks.append(piece_mask)
            seg_file_ids.append(item["file_id"])

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)}")

    mels = torch.from_numpy(np.stack(seg_mels))[:, None, :, :]  # (조각 수, 1, 80, 126)
    masks = torch.from_numpy(np.stack(seg_masks))[:, None, None, :]  # (조각 수, 1, 1, 126)

    torch.save({"mels": mels, "masks": masks, "file_ids": seg_file_ids}, val_out_path)
    print(f"[dev] 저장 완료: 조각 {mels.shape[0]}개, 건너뛴 파일 {skipped}개")
    print(f"  Mel 크기 {tuple(mels.shape)}, mask 크기 {tuple(masks.shape)}")
    print(f"  값 범위 {mels.min().item():.2f} ~ {mels.max().item():.2f}, 평균 {mels.mean().item():.3f}, 표준편차 {mels.std().item():.3f}")


def save_example(logmel, mask, out_path):  # 확인용 Mel 그림을 저장한다.

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    axes[0].imshow(logmel.numpy(), origin="lower", aspect="auto", cmap="magma")  # 아래쪽이 낮은 주파수
    axes[0].set_ylabel("Mel bin")
    axes[0].set_title("normalized log-Mel (train example)")
    axes[1].plot(mask.numpy())  # mask(1=사용, 0=제외)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel("mask")
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ===============================
# 6. 실행
# ===============================

if __name__ == "__main__":  # python preprocess.py로 실행했을 때만 아래가 동작한다.

    os.makedirs(PREPROCESS_DIR, exist_ok=True)  # 결과 폴더를 만든다.

    print("설정 확인")
    print(f"  학습 데이터: {TRAIN_DATA}, 검증 데이터: {DEV_DATA}")
    print(f"  2초 조각 = {SEGMENT_FRAMES}프레임, 1초 = {MIN_SEGMENT_FRAMES}프레임")
    print(f"  무음 mask: {MASK_SILENCE}, 파일 개수 제한: {FILE_LIMIT}")
    print(f"  입력 종류: {FEATURE_TYPE}, 세로 칸 수: {N_BINS}, 프레임 간격: {FRAME_HOP}샘플")
    print(f"  코덱 증강: {AUGMENT_CODEC} {AUGMENT_CODECS if AUGMENT_CODEC else ''}")

    if AUGMENT_CODEC:  # 증강한 소리를 직접 들어 볼 수 있게 예시를 먼저 저장한다. (ffmpeg가 제대로 동작하는지도 여기서 먼저 확인된다)
        example_dir = os.path.join(PREPROCESS_DIR, "codec_examples")
        save_codec_examples(select_bonafide(TRAIN_DATA)[0]["path"], example_dir)
        print(f"  코덱 예시 저장: {example_dir}")

    print(f"  환경음 증강: {AUGMENT_NOISE} {f'SNR {NOISE_SNR_RANGE}dB, 앞뒤 빈 구간 {NOISE_PAD_SEC_RANGE}초' if AUGMENT_NOISE else ''}")
    if AUGMENT_NOISE:  # 섞은 소리를 직접 들어 볼 수 있게 예시를 먼저 저장한다.
        example_dir = os.path.join(PREPROCESS_DIR, "noise_examples")
        save_noise_examples(select_bonafide(TRAIN_DATA)[0]["path"], example_dir)
        print(f"  환경음 예시 저장: {example_dir}")

    print(f"  방 환경 증강: {AUGMENT_ROOM} {f'RT60 {ROOM_RT60_RANGE}초, DRR {ROOM_DRR_RANGE}dB, SNR {ROOM_SNR_RANGE}dB' if AUGMENT_ROOM else ''}")
    if AUGMENT_ROOM:  # 입힌 소리를 직접 들어 볼 수 있게 예시를 먼저 저장한다.
        example_dir = os.path.join(PREPROCESS_DIR, "room_examples")
        save_room_examples(select_bonafide(TRAIN_DATA)[0]["path"], example_dir)
        print(f"  방 환경 예시 저장: {example_dir}")

    stats, example_mel, example_mask = build_train(
        stats_out_path=os.path.join(PREPROCESS_DIR, "mel_stats.pt"),
        train_out_path=os.path.join(PREPROCESS_DIR, "train_full_mels.pt"),
    )

    build_val(stats, val_out_path=os.path.join(PREPROCESS_DIR, "val_mels.pt"))

    save_example(example_mel, example_mask, os.path.join(PREPROCESS_DIR, "mel_example.png"))
    print(f"확인용 그림 저장: {os.path.join(PREPROCESS_DIR, 'mel_example.png')}")
    print("전처리 끝")
