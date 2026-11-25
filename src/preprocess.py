# ===============================
# preprocess.py
# 음성 파일 하나 → 앞뒤 무음 자름·크기 맞춤 → (학습 real은 증강본도) → XLS-R 22층 프레임 특징 → 프레임 평균
# = 파일 벡터 하나(1024)로 바꿔서 저장한다. 이 벡터가 디퓨전(train.py)의 입력이다.
# XLS-R은 학습하지 않고 숫자로 바꾸는 일만 한다.
#
# 만드는 파일 (outputs/rdm_features_xlsr_l22 폴더, 데이터·증강마다 하나. 이미 있으면 다시 만들지 않는다)
#   <데이터>_clean.npz   : 원본 그대로             (예: 2019_train_clean.npz, libri_dev_clean.npz)
#   <데이터>_aug.npz     : 파일마다 증강본 하나     (학습 real, AUGMENT일 때) 방 환경 또는 배속
#   <EVAL_DATA>_room.npz : eval에 방 환경을 입힌 것 (방에서 녹음한 소리에서도 버티는지 평가)
#   npz 안: vectors (파일 수, 1024), file_id, label, attack_id, speaker_id
#
# 증강·방 환경은 파일 ID로 난수를 정한다. → 언제 돌려도 같은 파일에 같은 증강이 입혀진다.
# load_audio, file_vector는 evaluate.py(시험지)와 웹(explain.py)에서도 그대로 쓴다. → 학습과 판정이 똑같은 벡터를 본다.
# ===============================

import os  # 파일 경로를 다룬다.
import subprocess  # ffmpeg 프로그램을 실행한다. (m4a 같은 압축 음성을 읽을 때)
import zlib  # 파일 ID → 난수 시작값 (crc32: 같은 글자면 늘 같은 숫자)

import imageio_ffmpeg  # venv 안에 설치된 ffmpeg 실행 파일의 위치를 알려 준다.
import librosa  # 음성 파일을 읽고, 무음을 자르고, 배속을 바꾼다.
import numpy as np  # 숫자 배열을 계산한다.
import torch  # XLS-R을 GPU에서 실행한다.
from scipy.signal import fftconvolve  # 방 울림 입히기 (긴 소리도 빠르게 합성곱)
from transformers import AutoFeatureExtractor, AutoModel  # Hugging Face에서 XLS-R을 불러온다.

from config import *  # config.py의 모든 설정값을 가져온다.


# ===============================
# 1. 파일 목록 (라벨 파일 또는 폴더)
# ===============================

def read_protocol(data_name):  # data_name: config.py의 DATASETS 이름 (예: "2019_train", "5_eval")

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


def read_folder(data_name):  # 라벨 파일 없이 폴더 안의 음성을 모두 real로 읽는다. (LibriSpeech) read_protocol과 같은 모양으로 돌려준다.

    dataset = DATASETS[data_name]  # 음성 폴더, 이 데이터에 넣을 폴더 이름 목록
    items = []

    for folder, sub_folders, files in os.walk(dataset["audio_dir"]):  # 음성 폴더 아래의 모든 폴더를 하나씩 본다.
        sub_folders.sort()  # 폴더를 이름순으로 돌아서, 언제 실행해도 파일 순서가 같게 한다.
        speaker_id = os.path.basename(folder)  # 폴더 이름을 화자 ID로 쓴다. (LibriSpeech: 화자 안의 챕터 폴더)
        if not any(key in speaker_id for key in dataset["speakers"]):  # 이 데이터의 폴더가 아니면 건너뛴다.
            continue
        for name in sorted(files):  # 파일 이름순으로
            if name.lower().endswith((".wav", ".flac")):
                items.append({"path": os.path.join(folder, name), "speaker_id": speaker_id,
                              "file_id": os.path.splitext(name)[0], "attack_id": "-", "label": "bonafide"})  # 모두 진짜 사람 음성

    if len(items) == 0:  # 하나도 못 찾았으면 폴더 경로가 잘못된 것이다.
        raise ValueError(f"{data_name}: 음성을 하나도 찾지 못했다. config.py의 audio_dir와 speakers를 확인할 것")
    print(f"[{data_name}] 폴더 {len({it['speaker_id'] for it in items})}개, 파일 {len(items)}개")

    return items


def select_bonafide(data_name):  # 해당 데이터에서 real(bonafide) 파일만 고른다. (real-only)

    if "speakers" in DATASETS[data_name]:  # 라벨 파일이 없는 폴더형 데이터 (LibriSpeech)
        return read_folder(data_name)

    return [item for item in read_protocol(data_name) if item["label"] == "bonafide"]  # ASVspoof: 라벨 파일에서 bonafide만


def select_eval_items():  # eval 파일 목록 (real + fake). EVAL_LIMIT가 있으면 무작위로 그 개수만 고른다.

    items = read_protocol(EVAL_DATA)  # bona fide + spoof 전체

    if EVAL_LIMIT is not None:
        generator = torch.Generator().manual_seed(SEED)  # 언제 돌려도 같은 파일들이 뽑히도록 고정 (지금까지의 8,000개 표본과 같다)
        picked = torch.randperm(len(items), generator=generator)[:EVAL_LIMIT]  # 무작위 순서에서 앞의 EVAL_LIMIT개
        items = [items[i] for i in sorted(picked.tolist())]  # 원래 순서대로 정렬

    return items


def select_extra_eval(name):  # EXTRA_EVAL 데이터: 라벨 파일 중 실제로 내려받아 둔, 코덱 없는 파일만

    have = set(os.listdir(DATASETS[name]["audio_dir"]))  # 폴더에 있는 파일 이름 (일부만 내려받았으므로)
    items = [it for it in read_protocol(name) if it["codec"] == "-" and os.path.basename(it["path"]) in have]
    print(f"[{name}] 코덱 없는 파일 {len(items)}개 (real {sum(it['label'] == 'bonafide' for it in items)})")

    return items


# ===============================
# 2. 음성 읽기 (무음 파일 거르기 + 앞뒤 무음 자르기 + 크기 맞추기)
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


def load_audio(path):  # 음성 파일 하나를 읽는다. 학습·검증·평가·시험지·웹 모두 이 함수를 거친다.

    if path.lower().endswith((".flac", ".wav")):  # ASVspoof(flac) 등 압축 없는 형식
        wave, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)  # 16kHz 모노로 읽는다. (이미 16kHz면 그대로)
    else:  # m4a, mp3 등 압축 형식 (시험지 폴더)
        wave = read_with_ffmpeg(path)

    peak = float(np.abs(wave).max()) if len(wave) > 0 else 0.0  # 원래 최대 진폭 (빈 파일이면 0)
    if peak < MIN_PEAK:  # 소리가 거의 없는 파일 (디지털 무음, 마이크 꺼짐)
        raise ValueError(f"소리가 없는 파일이다 (최대 진폭 {peak:.5f} < {MIN_PEAK}): {path}")

    wave, _ = trim_silence(wave)  # 1) 앞뒤 무음을 잘라 낸다. (무음 길이가 real/fake 단서가 되지 않도록)
    wave = normalize_peak(wave)  # 2) 소리 크기만 맞춘다. (ASVspoof는 이미 1.0이라 그대로)

    return wave.astype(np.float32)  # 32비트 실수로 돌려준다.


# ===============================
# 3. 증강: 방 환경, 배속 (load_audio를 거친 소리에 입힌다)
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


def file_rng(file_id, variant):  # 파일·증강 종류마다 고정된 난수 (언제 돌려도 같은 증강)

    return np.random.default_rng(zlib.crc32(f"{file_id}_{variant}".encode()))


# ===============================
# 4. XLS-R: 음성 → 파일 벡터
# ===============================

def load_xlsr():  # 사전학습 XLS-R과 입력 정리 도구를 불러온다. (처음 실행 때 약 1.2GB를 내려받고, 그 뒤로는 내 컴퓨터에서 읽는다)

    extractor = AutoFeatureExtractor.from_pretrained(XLSR_MODEL)  # waveform을 모델 입력 형태로 바꾸는 규칙 (정규화 여부 등)
    model = AutoModel.from_pretrained(XLSR_MODEL).to(DEVICE).eval()  # eval(): dropout을 끄고 늘 같은 출력을 내게 한다.

    return extractor, model


def frame_features(wave, extractor, model):  # load_audio를 거친 waveform → XLSR_LAYER층 프레임 특징 (프레임 수, 1024) torch

    if extractor.do_normalize:  # XLS-R은 사전학습 때 파일마다 평균 0, 표준편차 1로 맞춘 소리를 봤다. 똑같이 맞춰 넣는다.
        wave = (wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)

    x = torch.from_numpy(wave.astype(np.float32))[None].to(DEVICE)  # (1, 샘플 수): 배치 1개
    with torch.no_grad():  # 학습하지 않으므로 gradient를 계산하지 않는다. (메모리 절약)
        out = model(x, output_hidden_states=True)  # 모든 층의 출력을 받는다.

    # hidden_states: 0번 = Transformer에 들어가기 직전 값(CNN → 1024로 넓히기 → 위치 정보 더하기), 1~24번 = 각 Transformer 층의 출력
    return out.hidden_states[XLSR_LAYER][0]  # (프레임 수, 1024): 22층 출력. 20ms마다 벡터 하나


def pool(frames):  # 프레임 특징 (프레임 수, 1024) torch → 벡터 하나 (1024,) numpy. 파일 전체에도, 웹 설명용 창에도 같은 방법을 쓴다.

    return frames.mean(dim=0).float().cpu().numpy()  # 차원마다 프레임 평균 = "이 구간 전체의 성질". 시간 축이 사라진다.


def file_vector(wave, extractor, model):  # waveform → 파일 벡터 (1024,). 학습·평가·웹이 모두 이 함수를 쓴다.

    return pool(frame_features(wave, extractor, model))


# ===============================
# 5. 파일 목록 → 벡터 묶음 저장·불러오기
# ===============================

def build_split(items, name, variant, extractor, model):  # name: 데이터 이름, variant: "clean", "aug", "room"

    out_path = os.path.join(FEATURE_DIR, f"{name}_{variant}.npz")
    if os.path.exists(out_path):  # 이미 만들어 두었으면 다시 하지 않는다. (다시 만들려면 파일을 지운다)
        print(f"[{name}_{variant}] 이미 있음")
        return

    vectors, kinds = [], []  # 파일 벡터, 증강 종류 (확인용)
    for i, item in enumerate(items):
        wave = load_audio(item["path"])  # 16kHz 모노 → 무음 파일이면 오류 → 앞뒤 무음 자르기 → 최대 진폭 1.0
        if variant == "aug":  # 방 환경 또는 배속 중 하나
            wave, kind = augment(wave, file_rng(item["file_id"], variant))
            kinds.append(kind)
        elif variant == "room":  # 평가용 방 환경
            wave = add_room(wave, file_rng(item["file_id"], variant))
        vectors.append(file_vector(wave, extractor, model))
        if (i + 1) % 1000 == 0:  # 진행 상황
            print(f"  [{name}_{variant}] {i + 1}/{len(items)}")

    vectors = np.stack(vectors).astype(np.float32)  # (파일 수, 1024)
    np.savez(out_path, vectors=vectors,  # 벡터와 함께 파일 정보를 같은 순서로 저장한다.
             file_id=[it["file_id"] for it in items], label=[it["label"] for it in items],
             attack_id=[it["attack_id"] for it in items], speaker_id=[it["speaker_id"] for it in items])
    print(f"[{name}_{variant}] 저장: {vectors.shape}, 값 범위 {vectors.min():.2f} ~ {vectors.max():.2f}"
          + (f", 증강 {dict(zip(*np.unique(kinds, return_counts=True)))}" if kinds else ""))


def load_split(name, variant="clean"):  # 저장한 벡터 묶음 하나를 불러온다.

    data = np.load(os.path.join(FEATURE_DIR, f"{name}_{variant}.npz"))
    return {key: data[key] for key in data.files}  # vectors, file_id, label, attack_id, speaker_id


def load_many(names, variants=("clean",)):  # 여러 데이터·증강을 이어 붙여 불러온다. source: 각 파일이 어느 묶음에서 왔는지

    parts = []
    for name in names:
        for variant in variants:
            part = load_split(name, variant)
            part["source"] = np.full(len(part["file_id"]), f"{name}_{variant}")  # 예: "libri_train_aug"
            parts.append(part)

    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}  # 같은 열끼리 이어 붙인다.


# ===============================
# 6. 실행
# ===============================

if __name__ == "__main__":

    os.makedirs(FEATURE_DIR, exist_ok=True)
    extractor, model = load_xlsr()
    print(f"XLS-R: {XLSR_MODEL}, {XLSR_LAYER}층, 장치 {DEVICE}")

    for name in TRAIN_DATA:  # 학습 real: 원본 (+ 증강본)
        items = select_bonafide(name)
        build_split(items, name, "clean", extractor, model)
        if AUGMENT:
            build_split(items, name, "aug", extractor, model)
    for name in DEV_DATA:  # dev real: 원본 그대로
        build_split(select_bonafide(name), name, "clean", extractor, model)
    eval_items = select_eval_items()  # eval 8,000개 (real + fake)
    build_split(eval_items, EVAL_DATA, "clean", extractor, model)
    for name in EXTRA_EVAL:  # 최신 공격 확인용 (ASVspoof 5)
        build_split(select_extra_eval(name), name, "clean", extractor, model)
    build_split(eval_items, EVAL_DATA, "room", extractor, model)  # 같은 파일에 방 환경을 입힌 것 (가장 오래 걸려서 마지막에)
