# ===============================
# feature_maha.py
# 웹 판정용 "특징 거리" 점수: 음성 모델(XLS-R 22층) 특징의 파일 평균이 진짜 음성 무리에서 얼마나 떨어졌는지 (마할라노비스 거리)
#
# 왜 디퓨전 점수가 아니라 이것으로 판정하나
#   같은 XLS-R 22층에서 디퓨전 오차 점수는 2019 EER 22.76%, 이 거리 점수는 11.64% (무음 자름, eval 8,000개 표본).
#   디퓨전은 판정 대신 "어느 구간이 이상한지" 보여 주는 설명(explain.py의 시간축 그림)에 쓴다.
#   참고: 디퓨전 없이 특징 거리만 쓰는 방식은 MLN-WavLM(arXiv 2611.11465)과 같은 계열이다.
#
# 진짜 음성 무리 (fake는 쓰지 않는다)
#   train real 2,580개 + FEATURE_MAHA_ROOM이면 같은 파일에 방 환경(울림 + 핑크 노이즈)을 입힌 2,580개
#   → 방에서 녹음한 진짜 음성도 "진짜 범위"로 본다. (깨끗한 real로만 맞추면 방 환경 real 오탐 100%, 넣으면 0%)
#   임계값: dev real 파일 점수의 95백분위 (THRESHOLD_PERCENTILE)
#
# 사용법
#   python src/feature_maha.py      기준값(feature_reference.npz)을 만든다. 특징 모델·층이 바뀌면 다시 실행
#   explain.py가 불러서 웹 판정에 쓴다.
# ===============================

import os  # 경로
import time  # 기준값을 만든 시각
import zlib  # 파일 ID → 난수 시작값 (방 환경을 파일마다 고정)

import numpy as np  # 계산
from sklearn.covariance import LedoitWolf  # 차원(1024)이 많을 때도 안정적인 공분산

from config import *  # 경로, WAVLM_*, FEATURE_MAHA_*, THRESHOLD_PERCENTILE 등
from preprocess import load_audio, random_room, select_bonafide, wave_to_wavlm  # 학습과 같은 방식으로 읽고 특징을 뽑는다.

MAD_SCALE = 1.4826  # MAD(중앙값 절대 편차)를 표준편차와 같은 크기로 바꾸는 상수 (정규분포일 때)


# ===============================
# 1. 파일 하나 → 특징 평균
# ===============================

def pooled_feature(wave):  # waveform(무음 자름·크기 맞춤 끝난 것) → (WAVLM_DIM,): 고른 층 특징의 시간 평균

    return wave_to_wavlm(wave).mean(axis=1)  # (WAVLM_DIM, 프레임 수) → 프레임 평균


def collect(items, room=False):  # 파일 목록 → (파일 수, WAVLM_DIM). room이면 방 환경을 입힌 뒤 뽑는다.

    feats = []
    for i, item in enumerate(items):
        wave = load_audio(item["path"])
        if room:  # 파일 ID로 난수를 정해서 언제 돌려도 같은 방 환경이 입혀진다.
            wave = random_room(wave, np.random.default_rng(zlib.crc32(item["file_id"].encode())))
        feats.append(pooled_feature(wave))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)}")
    return np.stack(feats)


# ===============================
# 2. 진짜 음성 무리와의 거리
# ===============================

def distance(feats, ref):  # (N, D) 또는 (D,) → 마할라노비스 거리를 진짜 음성 기준으로 크기 맞춘 점수 (클수록 진짜답지 않다)

    z = (np.atleast_2d(feats) - ref["mean"]) / ref["std"]  # 차원마다 train real 기준으로 표준화
    diff = z - ref["location"]  # 진짜 음성 무리의 중심에서 얼마나 떨어졌나
    d = np.sqrt(np.maximum(np.einsum("nd,de,ne->n", diff, ref["precision"], diff), 0.0))  # √(차이ᵀ · 공분산⁻¹ · 차이)
    return (d - ref["median"]) / ref["mad"]  # train real 거리의 중앙값을 0, 흔들림(MAD)을 1로 맞춘다. (MLN-WavLM과 같다)


def load_reference():  # feature_reference.npz → dict (explain.py가 쓴다)

    with np.load(FEATURE_REFERENCE_PATH) as data:
        return {key: data[key] for key in data.files}


def build_reference():  # train real(+방 환경)으로 진짜 음성 무리를 정하고, dev real로 임계값을 정해 저장한다.

    print(f"특징: {WAVLM_MODEL_NAME} {WAVLM_LAYER}층, 방 환경 real 포함: {FEATURE_MAHA_ROOM}")
    train_items, dev_items = select_bonafide(TRAIN_DATA), select_bonafide(DEV_DATA)

    print(f"[{TRAIN_DATA}] real {len(train_items)}개")
    train = collect(train_items)
    if FEATURE_MAHA_ROOM:
        print(f"[{TRAIN_DATA}] 같은 파일에 방 환경을 입힌 {len(train_items)}개")
        train = np.concatenate([train, collect(train_items, room=True)])
    print(f"[{DEV_DATA}] real {len(dev_items)}개 (임계값용)")
    dev = collect(dev_items)

    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-8  # 표준화 값
    lw = LedoitWolf().fit((train - mean) / std)  # 진짜 음성 무리의 중심·공분산
    ref = {"mean": mean, "std": std, "location": lw.location_, "precision": lw.precision_, "median": 0.0, "mad": 1.0}
    raw = distance(train, ref)  # 크기를 맞추기 전 train 거리
    ref["median"] = float(np.median(raw))
    ref["mad"] = float(MAD_SCALE * np.median(np.abs(raw - ref["median"])))

    dev_scores = distance(dev, ref)
    ref["dev_scores"] = dev_scores
    ref["threshold"] = float(np.percentile(dev_scores, THRESHOLD_PERCENTILE))
    ref["meta"] = np.array(f"{WAVLM_MODEL_NAME}|{WAVLM_LAYER}|room={FEATURE_MAHA_ROOM}|train={len(train)}|dev={len(dev)}|{time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(os.path.dirname(FEATURE_REFERENCE_PATH), exist_ok=True)
    np.savez(FEATURE_REFERENCE_PATH, **ref)
    print(f"진짜 음성 무리: {len(train)}개, 임계값(dev real {THRESHOLD_PERCENTILE}%): {ref['threshold']:.4f}")
    print(f"저장: {FEATURE_REFERENCE_PATH}")


if __name__ == "__main__":
    build_reference()
