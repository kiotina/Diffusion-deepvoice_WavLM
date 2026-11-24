# ===============================
# preprocess.py
# 음성 파일을 디퓨전(U-Net)에 넣을 그림(특징)으로 바꿔서 저장한다.
# 특징의 종류는 config.py의 FEATURES로 고른다. 여러 개면 채널로 쌓는다. → 모양 (채널 수, 256칸, 프레임 수)
#
# 만드는 파일 (outputs/<RUN_NAME>/preprocess 폴더)
#   feature_stats.pt     : train real로 계산한 채널·주파수 칸별 평균·표준편차
#   train_features.pt    : train real 파일마다 전체 길이의 정규화된 특징 (학습 때 무작위로 2초씩 자른다)
#   val_features.pt      : dev real을 겹치지 않는 2초 조각으로 자른 정규화된 특징 (검증용)
#   feature_example.png  : 확인용 그림 (채널마다 한 장)
#
# 이 파일의 함수(load_audio, file_to_feature, cut_segments 등)는 evaluate.py에서도 그대로 가져다 쓴다.
# → 학습과 평가가 항상 똑같은 전처리를 거친다.
# ===============================

import os  # 파일 경로를 다룬다.
import subprocess  # ffmpeg 프로그램을 실행한다. (m4a 같은 압축 음성을 읽을 때)

import imageio_ffmpeg  # venv 안에 설치된 ffmpeg 실행 파일의 위치를 알려 준다.
import librosa  # 음성 파일을 읽고 STFT를 계산한다.
import matplotlib  # 그림을 저장한다.
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt  # 그림을 그린다.
import numpy as np  # 숫자 배열을 계산한다.
import torch  # 결과를 PyTorch 텐서로 저장한다.
from scipy.signal import fftconvolve  # 방 울림 입히기 (긴 소리도 빠르게 합성곱)

from config import *  # config.py의 모든 설정값을 가져온다.


# ===============================
# 1. 라벨(프로토콜) 읽기
# ===============================

def read_protocol(data_name):  # data_name: config.py의 DATASETS 이름 (예: "2019_train", "2021_eval")

    dataset = DATASETS[data_name]  # 라벨 파일, 음성 폴더, 칸 번호
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
                "codec": parts[columns["codec"]] if "codec" in columns else "-",  # 거친 코덱 (ASVspoof 5만 있다. "-" = 없음)
            })

    if len(items) == 0:  # 한 줄도 못 읽었으면 칸 번호나 라벨 파일이 잘못된 것이다. (조용히 0개로 넘어가지 않도록)
        raise ValueError(f"{data_name}: 라벨을 한 줄도 읽지 못했다. config.py의 protocol 경로와 columns를 확인할 것")

    return items  # 파일 정보 목록을 돌려준다.


def read_folder(data_name):  # 라벨 파일 없이 폴더 안의 wav를 모두 real로 읽는다. (한국어 데이터용) read_protocol과 같은 모양으로 돌려준다.

    dataset = DATASETS[data_name]  # 음성 폴더, 이 데이터에 넣을 화자 목록
    items = []

    for folder, sub_folders, files in os.walk(dataset["audio_dir"]):  # 음성 폴더 아래의 모든 폴더를 하나씩 본다.
        sub_folders.sort()  # 폴더를 이름순으로 돌아서, 언제 실행해도 파일 순서가 같게 한다.
        speaker_id = os.path.basename(folder)  # 폴더 이름을 화자 ID로 쓴다. (한국어: 화자 폴더, LibriSpeech: 화자 안의 챕터 폴더)
        if not any(key in speaker_id for key in dataset["speakers"]):  # 이 데이터의 화자가 아니면 건너뛴다. (train 화자가 dev에 섞이지 않게)
            continue
        for name in sorted(files):  # 파일 이름순으로
            if name.lower().endswith((".wav", ".flac")):  # 한국어는 wav, LibriSpeech는 flac
                items.append({"path": os.path.join(folder, name), "speaker_id": speaker_id,
                              "file_id": os.path.splitext(name)[0], "attack_id": "-", "label": "bonafide"})  # 모두 진짜 사람 음성

    if len(items) == 0:  # 하나도 못 찾았으면 폴더 경로나 화자 이름이 잘못된 것이다.
        raise ValueError(f"{data_name}: wav를 하나도 찾지 못했다. config.py의 audio_dir와 speakers를 확인할 것")
    print(f"[{data_name}] 화자 {len({it['speaker_id'] for it in items})}명, 파일 {len(items)}개")

    return items


def select_bonafide(data_name):  # 해당 데이터에서 real(bonafide) 파일만 고른다. (real-only)

    if "speakers" in DATASETS[data_name]:  # 라벨 파일이 없는 폴더형 데이터 (한국어)
        items = read_folder(data_name)
    else:  # ASVspoof: 라벨 파일에서 bonafide만
        items = [item for item in read_protocol(data_name) if item["label"] == "bonafide"]

    if FILE_LIMIT is not None:  # 개발용으로 개수를 줄인 경우
        items = items[:FILE_LIMIT]  # 앞에서부터 FILE_LIMIT개만 사용한다.

    return items


# ===============================
# 2. 음성 읽기 (무음 파일 거르기 + 무음 자르기 + 크기 맞추기)
# ===============================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()  # venv 안의 ffmpeg 실행 파일 경로


def read_with_ffmpeg(path):  # m4a·mp3 같은 압축 음성 → 16kHz 모노 waveform (librosa가 m4a를 못 읽어서)

    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",  # 진행 메시지는 숨기고 오류만 보인다.
         "-i", path,  # 이 파일을 읽어서
         "-ac", "1", "-ar", str(SAMPLE_RATE),  # 모노, 16kHz로 바꾸고
         "-f", "f32le", "pipe:1"],  # 32비트 실수 숫자 그대로 돌려받는다. (파일을 따로 만들지 않는다)
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,  # ffmpeg가 실패하면 바로 오류를 낸다. (조용히 빈 소리로 넘어가지 않도록)
    )

    return np.frombuffer(result.stdout, dtype=np.float32).copy()  # bytes → 숫자 배열 (copy: 고칠 수 있는 배열로)


def trim_silence(wave):  # 앞뒤 무음을 잘라 낸다. 반환: (잘린 소리, 앞에서 잘라 낸 길이(샘플))

    trimmed, (start, _) = librosa.effects.trim(  # 가장 큰 소리보다 40dB 이상 작은 부분을 앞뒤에서만 잘라 낸다. (가운데 쉼은 그대로)
        wave,
        top_db=-SILENCE_DB_THRESHOLD,  # 40
        frame_length=WIN_LENGTH,  # 25ms 창으로 소리 크기를 잰다.
        hop_length=WIN_LENGTH // 2,  # 12.5ms씩 옮긴다.
    )

    return trimmed, int(start)  # start: 나중에 원래 녹음과 시간을 맞출 때 쓴다. (웹)


def normalize_peak(wave):  # 가장 큰 진폭이 1.0이 되도록 전체에 같은 숫자를 곱한다.

    return wave / float(np.abs(wave).max())  # load_audio에서 무음 파일을 먼저 걸러 내므로 0으로 나눌 일은 없다.


def load_audio(path):  # 음성 파일 하나를 읽는다. 학습·검증·평가·시험지 모두 이 함수를 거친다.

    if path.lower().endswith((".flac", ".wav")):  # ASVspoof(flac) 등 압축 없는 형식
        wave, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)  # 16kHz 모노로 읽는다. (이미 16kHz면 그대로)
    else:  # m4a, mp3 등 압축 형식 (시험지 폴더)
        wave = read_with_ffmpeg(path)

    peak = float(np.abs(wave).max()) if len(wave) > 0 else 0.0  # 원래 최대 진폭 (빈 파일이면 0)
    if peak < MIN_PEAK:  # 소리가 거의 없는 파일 (디지털 무음, 마이크 꺼짐)
        raise ValueError(f"소리가 없는 파일이다 (최대 진폭 {peak:.5f} < {MIN_PEAK}): {path}")

    wave, _ = trim_silence(wave)  # 1) 앞뒤 무음을 잘라 낸다. (무음 길이가 real/fake 단서가 되지 않도록)
    wave = normalize_peak(wave)  # 2) 소리 크기만 맞춘다. (ASVspoof는 이미 1.0이라 그대로)
    # 파일별 스펙트로그램 정규화 같은 "음색을 바꾸는" 정규화는 하지 않는다. 합성 흔적을 지울 수 있어서.

    return wave.astype(np.float32)  # 32비트 실수로 돌려준다.


# ===============================
# 2-1. 증강: 방 환경, 배속 (load_audio를 거친 소리에 입힌다)
# ===============================

def pink_noise(length, rng):  # 핑크 노이즈: 저음이 크고 고음으로 갈수록 약해지는 잡음 (방·마이크의 바닥 잡음과 비슷)

    spectrum = np.fft.rfft(rng.standard_normal(length))  # 화이트 노이즈(모든 주파수가 같은 세기)를 주파수로 바꾼다.
    freqs = np.fft.rfftfreq(length, d=1 / SAMPLE_RATE)  # 각 칸의 주파수(Hz)
    keep = freqs >= ROOM_NOISE_LOW_HZ  # 귀에 거의 안 들리는 초저음은 뺀다.
    spectrum[keep] /= np.sqrt(freqs[keep])  # 세기가 1/주파수가 되도록 → 진폭은 1/√주파수
    spectrum[~keep] = 0
    noise = np.fft.irfft(spectrum, length)  # 다시 소리로

    return noise / np.sqrt(np.mean(noise ** 2))  # 크기(RMS)를 1로 맞춘다. 실제 크기는 add_room에서 SNR로 정한다.


def room_impulse(rt60, drr_db, rng):  # 방의 잔향 응답: "손뼉을 한 번 쳤을 때 마이크에 들리는 소리"

    t = np.arange(1, int(rt60 * SAMPLE_RATE)) / SAMPLE_RATE  # 울림이 이어지는 시간 (직접 들어온 소리 바로 다음부터 RT60까지)
    tail = rng.standard_normal(len(t)) * np.exp(-np.log(1000) * t / rt60)  # 무작위 반사음이 점점 작아진다. RT60초 뒤에 1/1000 = -60dB
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail ** 2))  # 울림 전체의 에너지를 "직접 들어온 소리(1)보다 DRR dB 작게" 맞춘다.

    return np.concatenate([[1.0], tail])  # 맨 앞 1 = 입에서 마이크로 바로 들어온 소리, 그 뒤 = 벽에 반사된 울림


def add_room(wave, rng):  # 방 울림 → 앞뒤에 잡음만 있는 구간 → 핑크 노이즈로 전체 덮기. 값은 config 범위에서 무작위로

    rt60, drr_db, snr_db = (rng.uniform(*r) for r in (ROOM_RT60_RANGE, ROOM_DRR_RANGE, ROOM_SNR_RANGE))
    wet = fftconvolve(wave, room_impulse(rt60, drr_db, rng))  # (FFT로 빠르게) 말소리에 방 울림을 입힌다. (끝에 RT60만큼 울림 꼬리가 붙는다)
    pads = [np.zeros(int(rng.uniform(*ROOM_PAD_SEC_RANGE) * SAMPLE_RATE)) for _ in range(2)]  # 앞, 뒤 빈 구간 (아래에서 잡음으로 덮인다)
    padded = np.concatenate([pads[0], wet, pads[1]])
    noise = pink_noise(len(padded), rng) * np.sqrt(np.mean(wet ** 2)) / 10 ** (snr_db / 20)  # 말소리 크기보다 SNR dB만큼 작은 바닥 잡음

    return normalize_peak(padded + noise).astype(np.float32)  # 크기를 다시 1.0으로 (실제 녹음도 load_audio에서 이렇게 맞춘다)


def change_speed(wave, speed):  # 말 속도·목소리 높이를 함께 바꾼다. speed 1.2 = 1.2배 빠르고 약 3음 높게

    # 16kHz 소리를 16k/speed 샘플로 다시 뽑은 뒤 그대로 16kHz로 재생한다. (테이프 빨리 감기)
    # 피치만 바꾸는 방법은 소리를 다시 합성하면서 합성 흔적 같은 왜곡을 만들어서 쓰지 않는다.
    return librosa.resample(wave, orig_sr=SAMPLE_RATE, target_sr=int(round(SAMPLE_RATE / speed)), res_type="soxr_hq").astype(np.float32)


def augment(wave, rng):  # 증강본 하나: 방 환경 또는 배속(0.9, 1.1, 1.2) 중 무작위 하나. 반환: (소리, 이름)

    choice = rng.integers(1 + len(AUGMENT_SPEEDS))  # 0 = 방 환경, 1~3 = 배속
    if choice == 0:
        return add_room(wave, rng), "room"
    speed = AUGMENT_SPEEDS[choice - 1]
    return change_speed(wave, speed), f"speed{speed}"


# ===============================
# 3. 음성 → 특징 (FEATURES를 채널로 쌓는다)
# ===============================

def spec_channel(stft):  # log 스펙트로그램: 주파수마다 소리의 세기

    power = np.abs(stft) ** 2  # 크기의 제곱(파워)
    return np.log(power + LOG_EPS)  # 값의 차이가 너무 크지 않도록 log로 압축한다.


def phase_channel(stft):  # 위상 변화 편차: 주파수마다 "다음 프레임에서 위상이 예상만큼 돌았는가"

    phase = np.angle(stft)  # 위상 (-π ~ π). 위상 그 자체는 프레임마다 마구 바뀌어 보여서 그대로는 쓸모가 없다.
    advance = np.diff(phase, axis=1, prepend=phase[:, :1])  # 프레임 사이에 위상이 돈 양. 첫 프레임은 앞이 없어서 0
    # 주파수 k칸의 순수한 소리는 한 프레임(HOP_LENGTH 샘플) 동안 위상이 정확히 2π·k·HOP/N_FFT만큼 돈다.
    expected = 2 * np.pi * np.arange(stft.shape[0])[:, None] * HOP_LENGTH / N_FFT  # (257, 1): 칸마다 예상 회전량
    deviation = np.angle(np.exp(1j * (advance - expected)))  # (실제 − 예상)을 -π ~ π로 감는다. (2π 차이는 같은 각도라서)
    deviation[:, 0] = 0.0  # 첫 프레임은 비교할 앞 프레임이 없다.
    # 편차가 0 근처 = 그 칸의 주파수가 칸 중심과 같고 안정적으로 이어진다. 편차가 크거나 들쭉날쭉 = 소리가 매끄럽게 이어지지 않는다.
    # 주의: 소리가 거의 없는 칸(고요한 고음 등)의 위상은 의미 없는 잡음이라 편차도 무작위로 나온다.

    return deviation


def residual_channel(stft):  # 잔차 스펙트로그램: log 스펙트로그램 − 매끈한 포락선(입 모양)

    log_power = spec_channel(stft)  # (257, 프레임 수)
    cepstrum = np.fft.irfft(log_power, axis=0)  # 켑스트럼: log 스펙트럼을 한 번 더 푸리에 변환 → "스펙트럼이 얼마나 빨리 출렁이는지"별로 나눈다. (512, 프레임 수)
    cepstrum[CEPSTRUM_LIFTER:-CEPSTRUM_LIFTER + 1] = 0.0  # 빠른 출렁임(배음 = 목소리 높이)을 지우고 느린 출렁임(포먼트)만 남긴다. (양쪽이 대칭이라 앞뒤 모두 남긴다)
    envelope = np.fft.rfft(cepstrum, axis=0).real  # 다시 log 스펙트럼으로 → 매끈한 포락선 (257, 프레임 수)

    return log_power - envelope  # 포락선을 빼면 음원의 미세 구조(배음, 잡음 성분)만 남는다.


def delta_channel(stft):  # 시간 변화량: 이웃 프레임과의 log 스펙트로그램 차이

    log_power = spec_channel(stft)  # (257, 프레임 수)
    return np.diff(log_power, axis=1, prepend=log_power[:, :1])  # 지금 프레임 − 앞 프레임. 첫 프레임은 0


CHANNEL_FUNCTIONS = {"spec": spec_channel, "phase": phase_channel, "residual": residual_channel, "delta": delta_channel}  # 특징 이름 → 만드는 함수


def wave_to_feature(wave):  # waveform → 특징 (채널 수, N_BINS, 프레임 수)

    stft = librosa.stft(  # 프레임마다 주파수 성분(복소수 = 세기 + 위상)을 계산한다. (257, 프레임 수)
        y=wave,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
    )
    channels = [CHANNEL_FUNCTIONS[name](stft)[:N_BINS] for name in FEATURES]  # 특징마다 257칸으로 계산한 뒤 맨 위 한 칸(8000Hz)을 버려 (256, 프레임 수)
    return np.stack(channels).astype(np.float32)  # (채널 수, 256, 프레임 수)


def normalize_feature(feature, stats):  # train 통계로 정규화한다. (채널·칸마다 평균 0, 표준편차 1 근처)

    mean = stats["mean"][:, :, None]  # (채널 수, 256, 1)
    std = stats["std"][:, :, None]  # (채널 수, 256, 1)

    return (feature - mean) / std  # 모든 파일에 같은 통계를 쓴다. (파일별 정규화는 하지 않는다)


def file_to_feature(path, stats):  # 파일 하나 → 정규화된 특징 (채널 수, 256, 프레임 수)

    return normalize_feature(wave_to_feature(load_audio(path)), stats)


# ===============================
# 4. 2초 조각으로 자르기
# ===============================

def pad_to_segment(piece):  # SEGMENT_FRAMES보다 짧은 조각의 오른쪽을 채운다. 반환: (조각, mask)

    num_frames = piece.shape[-1]  # 실제 프레임 수
    missing = SEGMENT_FRAMES - num_frames  # 부족한 프레임 수

    mask = np.ones(SEGMENT_FRAMES, dtype=np.float32)  # 1 = 진짜 소리, 0 = 채운 칸
    if missing > 0:  # 부족하면
        piece = np.pad(piece, ((0, 0), (0, 0), (0, missing)), constant_values=0.0)  # 시간 방향 오른쪽만 정규화된 값 0(평균)으로 채운다.
        mask[num_frames:] = 0.0  # 채운 칸은 mask=0 → loss와 점수에서 빠진다.

    return piece, mask  # (채널 수, 256, 126), (126,)


def cut_segments(feature, hop_frames):  # 정규화된 특징을 2초 조각들로 자른다. 반환: [(조각, mask, 시작 프레임), ...]

    segments = []
    total_frames = feature.shape[-1]  # 전체 프레임 수

    for start in range(0, total_frames, hop_frames):  # hop_frames 간격으로 시작 위치를 옮긴다.
        end = start + SEGMENT_FRAMES  # 조각의 끝 위치
        piece_len = min(end, total_frames) - start  # 실제로 남은 길이

        if piece_len < MIN_SEGMENT_FRAMES:  # 1초보다 짧게 남았으면
            break  # 버리고 끝낸다.

        piece, mask = pad_to_segment(feature[:, :, start:end])  # 부족하면 채운다.
        segments.append((piece, mask, start))

        if end >= total_frames:  # 파일 끝까지 다 잘랐으면
            break

    return segments


# ===============================
# 5. 정규화 통계 (train real만 사용)
# ===============================

def compute_stats(features):  # 채널·주파수 칸마다 평균과 표준편차를 구한다.

    total_sum = np.zeros((N_CHANNELS, N_BINS), dtype=np.float64)  # 칸별 합 (넘치지 않게 float64)
    total_sq_sum = np.zeros((N_CHANNELS, N_BINS), dtype=np.float64)  # 칸별 제곱합
    total_count = 0  # 사용한 프레임 수

    for feature in features:  # 파일마다 (채널 수, 256, 프레임 수)
        feature = feature.astype(np.float64)  # 합을 정확하게 내려고 float64로
        total_sum += feature.sum(axis=2)  # 시간 방향으로 더한다.
        total_sq_sum += (feature ** 2).sum(axis=2)
        total_count += feature.shape[2]

    mean = total_sum / total_count  # 평균
    std = np.sqrt(total_sq_sum / total_count - mean ** 2)  # 표준편차 = √(제곱의 평균 − 평균의 제곱)
    std = np.maximum(std, 1e-5)  # 0으로 나누지 않도록 아주 작은 값은 막는다.

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}  # 각각 (채널 수, 256)


def load_stats():  # 저장한 통계를 불러온다. (evaluate.py에서 쓴다)

    stats = torch.load(os.path.join(PREPROCESS_DIR, "feature_stats.pt"))
    return {"mean": stats["mean"].numpy(), "std": stats["std"].numpy()}  # 위 함수들이 numpy를 쓰므로 바꿔 준다.


# ===============================
# 6. 분할별 처리
# ===============================

def build_train():  # train real → 통계 + 파일별 전체 길이 특징

    items = select_bonafide(TRAIN_DATA)
    print(f"[train] real 파일 {len(items)}개 처리 시작")

    features, file_ids = [], []  # 정규화 전 특징, 파일 ID
    for i, item in enumerate(items):  # 파일마다
        features.append(wave_to_feature(load_audio(item["path"])))  # (채널 수, 256, 프레임 수)
        file_ids.append(item["file_id"])
        if (i + 1) % 500 == 0:  # 500개마다 진행 상황
            print(f"  {i + 1}/{len(items)}")

    stats = compute_stats(features)  # train real로만 통계를 계산한다.
    torch.save({"mean": torch.from_numpy(stats["mean"]), "std": torch.from_numpy(stats["std"])},
               os.path.join(PREPROCESS_DIR, "feature_stats.pt"))

    normalized = [torch.from_numpy(normalize_feature(f, stats)) for f in features]  # 정규화한 뒤 텐서로
    torch.save({"features": normalized, "file_ids": file_ids},  # 파일마다 길이가 달라서 목록(list)으로 저장한다.
               os.path.join(PREPROCESS_DIR, "train_features.pt"))

    total_frames = sum(f.shape[-1] for f in normalized)
    print(f"[train] 저장 완료: 파일 {len(normalized)}개, 전체 {total_frames}프레임 (약 {total_frames * HOP_LENGTH / SAMPLE_RATE / 60:.1f}분)")
    for c, name in enumerate(FEATURES):  # 채널마다 통계 범위를 확인한다.
        print(f"  {name} 통계: 평균 {stats['mean'][c].min():.2f} ~ {stats['mean'][c].max():.2f}, 표준편차 {stats['std'][c].min():.2f} ~ {stats['std'][c].max():.2f}")

    return stats, normalized[0]  # 확인용 그림에 쓸 첫 파일도 돌려준다.


def build_val(stats):  # dev real → 겹치지 않는 고정 2초 조각

    items = select_bonafide(DEV_DATA)
    print(f"[dev] real 파일 {len(items)}개 처리 시작")

    pieces, masks, file_ids = [], [], []  # 조각, mask, 조각의 원본 파일 ID
    skipped = 0  # 1초보다 짧아서 조각이 하나도 안 나온 파일 수

    for i, item in enumerate(items):
        segments = cut_segments(file_to_feature(item["path"], stats), hop_frames=SEGMENT_FRAMES)  # 겹치지 않게 2초씩

        if len(segments) == 0:
            skipped += 1
            continue

        for piece, mask, _ in segments:
            pieces.append(piece)
            masks.append(mask)
            file_ids.append(item["file_id"])

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)}")

    pieces = torch.from_numpy(np.stack(pieces))  # (조각 수, 채널 수, 256, 126)
    masks = torch.from_numpy(np.stack(masks))[:, None, None, :]  # (조각 수, 1, 1, 126): 채널·주파수 방향으로 늘려 쓴다.

    torch.save({"features": pieces, "masks": masks, "file_ids": file_ids}, os.path.join(PREPROCESS_DIR, "val_features.pt"))
    print(f"[dev] 저장 완료: 조각 {pieces.shape[0]}개, 건너뛴 파일 {skipped}개")
    print(f"  특징 크기 {tuple(pieces.shape)}, mask 크기 {tuple(masks.shape)}")
    for c, name in enumerate(FEATURES):  # train 통계로 정규화했으므로 dev도 평균 0, 표준편차 1 근처여야 한다. 크게 다르면 전처리가 어긋난 것이다.
        channel = pieces[:, c]
        print(f"  {name}: 값 범위 {channel.min().item():.2f} ~ {channel.max().item():.2f}, 평균 {channel.mean().item():.3f}, 표준편차 {channel.std().item():.3f}")


def save_example(feature, out_path):  # 확인용 그림을 저장한다. (채널마다 한 장씩 위아래로)

    fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(10, 3.5 * N_CHANNELS), squeeze=False)
    for c, name in enumerate(FEATURES):
        ax = axes[c, 0]
        image = ax.imshow(feature[c].numpy(), origin="lower", aspect="auto", cmap="magma")  # 아래쪽이 낮은 주파수
        ax.set_ylabel("bin (31.25Hz)")
        ax.set_title(f"normalized {name} (train example)")
        fig.colorbar(image, ax=ax)
    axes[-1, 0].set_xlabel("frame (16ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ===============================
# 7. 실행
# ===============================

if __name__ == "__main__":  # python src/preprocess.py로 실행했을 때만 아래가 동작한다.

    os.makedirs(PREPROCESS_DIR, exist_ok=True)  # 결과 폴더를 만든다.

    print("설정 확인")
    print(f"  실험 이름: {RUN_NAME}, 학습: {TRAIN_DATA}, 검증: {DEV_DATA}")
    print(f"  특징: {FEATURES} → 입력 크기 ({N_CHANNELS}, {N_BINS}, {SEGMENT_FRAMES})")
    print(f"  앞뒤 무음 자르기 기준: {SILENCE_DB_THRESHOLD}dB, 파일 개수 제한: {FILE_LIMIT}")

    stats, example = build_train()
    build_val(stats)

    example_path = os.path.join(PREPROCESS_DIR, "feature_example.png")
    save_example(example, example_path)
    print(f"확인용 그림 저장: {example_path}")
    print("전처리 끝")
