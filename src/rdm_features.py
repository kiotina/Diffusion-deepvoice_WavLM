# ===============================
# rdm_features.py
# 음성 파일 하나 → XLS-R 22층 특징 → 프레임 평균(+ 표준편차) = 파일 벡터 하나(1024 또는 2048)로 바꿔서 저장한다. (RDM_POOLING)
# 이 벡터가 RDM 디퓨전(rdm_train.py)의 입력이다. XLS-R은 학습하지 않고 숫자로 바꾸는 일만 한다.
#
# 만드는 파일 (outputs/rdm_features_xlsr_l22 폴더, 데이터·증강마다 하나. 이미 있으면 다시 만들지 않는다)
#   <데이터>_clean.npz : 원본 그대로             (예: 2019_train_clean.npz, ko_dev_clean.npz)
#   <데이터>_aug.npz   : 파일마다 증강본 하나     (학습 real, RDM_AUGMENT일 때) 방 환경 또는 배속
#   <EVAL_DATA>_room.npz : eval에 방 환경을 입힌 것 (방에서 녹음한 소리에서도 버티는지 평가)
#   npz 안: vectors (파일 수, 1024), file_id, label, attack_id, speaker_id
#
# 증강·방 환경은 파일 ID로 난수를 정한다. → 언제 돌려도 같은 파일에 같은 증강이 입혀진다.
# file_vector 함수는 rdm_evaluate.py(시험지)와 웹 서버에서도 그대로 쓴다. → 학습과 판정이 똑같은 벡터를 본다.
# ===============================

import os  # 파일 경로를 다룬다.
import zlib  # 파일 ID → 난수 시작값 (crc32: 같은 글자면 늘 같은 숫자)

import numpy as np  # 벡터를 저장한다.
import torch  # XLS-R을 GPU에서 실행한다.
from transformers import AutoFeatureExtractor, AutoModel  # Hugging Face에서 XLS-R을 불러온다.

from config import *  # config.py의 모든 설정값을 가져온다.
from evaluate import select_eval_items  # 지금까지 비교에 쓴 eval 8,000개 표본 (seed 고정)
from preprocess import add_room, augment, load_audio, read_protocol, select_bonafide  # 음성 읽기와 증강 (다른 실험과 똑같이)


# ===============================
# 1. XLS-R 불러오기
# ===============================

def load_xlsr():  # 사전학습 XLS-R과 입력 정리 도구를 불러온다. (처음 실행 때 약 1.2GB를 내려받는다)

    extractor = AutoFeatureExtractor.from_pretrained(XLSR_MODEL)  # waveform을 모델 입력 형태로 바꾸는 규칙 (정규화 여부 등)
    model = AutoModel.from_pretrained(XLSR_MODEL).to(DEVICE).eval()  # eval(): dropout을 끄고 늘 같은 출력을 내게 한다.

    return extractor, model


# ===============================
# 2. 음성 하나 → 파일 벡터
# ===============================

def file_vector(wave, extractor, model):  # load_audio를 거친 waveform → XLSR_LAYER층 프레임 평균 (1024,)

    if extractor.do_normalize:  # XLS-R은 사전학습 때 파일마다 평균 0, 표준편차 1로 맞춘 소리를 봤다. 똑같이 맞춰 넣는다.
        wave = (wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)

    x = torch.from_numpy(wave.astype(np.float32))[None].to(DEVICE)  # (1, 샘플 수): 배치 1개
    with torch.no_grad():  # 학습하지 않으므로 gradient를 계산하지 않는다. (메모리 절약)
        out = model(x, output_hidden_states=True)  # 모든 층의 출력을 받는다.
    frames = out.hidden_states[XLSR_LAYER][0]  # (프레임 수, 1024): 22층 출력. 20ms마다 벡터 하나 (0번은 CNN 출력, 1~24번이 Transformer 층)

    mean = frames.mean(dim=0)  # (1024,): 프레임 평균 = "이 파일 전체의 성질"
    if RDM_POOLING == "mean_std":  # 표준편차도 붙인다 → (2048,)
        std = frames.std(dim=0)  # (1024,): 차원마다 프레임끼리 얼마나 흔들리나 (작을수록 "매끈함")
        return torch.cat([mean, std]).float().cpu().numpy()

    return mean.float().cpu().numpy()


def select_extra_eval(name):  # RDM_EXTRA_EVAL 데이터: 라벨 파일 중 실제로 내려받아 둔, 코덱 없는 파일만

    have = set(os.listdir(DATASETS[name]["audio_dir"]))  # 폴더에 있는 파일 이름 (일부만 내려받았으므로)
    items = [it for it in read_protocol(name) if it["codec"] == "-" and os.path.basename(it["path"]) in have]
    print(f"[{name}] 코덱 없는 파일 {len(items)}개 (real {sum(it['label'] == 'bonafide' for it in items)})")

    return items


def file_rng(file_id, variant):  # 파일·증강 종류마다 고정된 난수 (언제 돌려도 같은 증강)

    return np.random.default_rng(zlib.crc32(f"{file_id}_{variant}".encode()))


# ===============================
# 3. 파일 목록 → 벡터 묶음 저장
# ===============================

def build_split(items, name, variant, extractor, model):  # name: 데이터 이름, variant: "clean", "aug", "room"

    out_path = os.path.join(RDM_FEATURE_DIR, f"{name}_{variant}.npz")
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

    data = np.load(os.path.join(RDM_FEATURE_DIR, f"{name}_{variant}.npz"))
    return {key: data[key] for key in data.files}  # vectors, file_id, label, attack_id, speaker_id


def load_many(names, variants=("clean",)):  # 여러 데이터·증강을 이어 붙여 불러온다. source: 각 파일이 어느 묶음에서 왔는지

    parts = []
    for name in names:
        for variant in variants:
            part = load_split(name, variant)
            part["source"] = np.full(len(part["file_id"]), f"{name}_{variant}")  # 예: "ko_train_aug"
            parts.append(part)

    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}  # 같은 열끼리 이어 붙인다.


# ===============================
# 4. 실행
# ===============================

if __name__ == "__main__":

    os.makedirs(RDM_FEATURE_DIR, exist_ok=True)
    extractor, model = load_xlsr()
    print(f"XLS-R: {XLSR_MODEL}, {XLSR_LAYER}층, 장치 {DEVICE}")

    for name in RDM_TRAIN_DATA:  # 학습 real: 원본 (+ 증강본)
        items = select_bonafide(name)
        build_split(items, name, "clean", extractor, model)
        if RDM_AUGMENT:
            build_split(items, name, "aug", extractor, model)
    for name in RDM_DEV_DATA:  # dev real: 원본 그대로
        build_split(select_bonafide(name), name, "clean", extractor, model)
    eval_items = select_eval_items()  # eval 8,000개 (real + fake)
    build_split(eval_items, EVAL_DATA, "clean", extractor, model)
    for name in RDM_EXTRA_EVAL:  # 최신 공격 확인용 (ASVspoof 5)
        build_split(select_extra_eval(name), name, "clean", extractor, model)
    build_split(eval_items, EVAL_DATA, "room", extractor, model)  # 같은 파일에 방 환경을 입힌 것 (가장 오래 걸려서 마지막에)
