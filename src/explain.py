# ===============================
# explain.py
# 최종 모델(WavLM 특징 + 디퓨전 U-Net)로 음성 하나를 분석해 "판정"과 "설명(XAI)"을 만든다.
#
# 설명 세 가지. 모두 점수를 만든 값 그대로를 나눠 보여 준다. (Grad-CAM 같은 사후 설명이 아니다)
#   1. 시간   : 프레임(WavLM 20ms)마다 노이즈 예측 오차가 dev real보다 얼마나 큰지 (z-점수)
#   2. t      : t=100/300/500 중 어느 노이즈 단계에서 real과 달랐는지, 오차가 컸는지 작았는지
#   3. 분포   : score_maha(Mahalanobis 거리)에 어떤 특징(t)이 얼마나 기여했는지
#
# WavLM의 768차원은 주파수가 아니라서 "어느 주파수가 이상한가"는 말할 수 없다.
# 화면의 Mel 그림은 사람이 위치를 찾기 쉽도록 까는 배경일 뿐, 모델 입력이 아니다.
#
# 사용법
#   python src/explain.py               dev real로 기준값을 만든다. (best.pt가 바뀔 때마다 한 번)
#   python src/explain.py 음성파일경로     파일 하나를 분석해 요약을 출력한다. (웹 없이 확인용)
#
# 만드는 파일 (outputs/<RUN_NAME>/explain 폴더)
#   reference.json   : 설명용 기준값. 모두 dev real로 계산한다. (real-only)
#     seg_calibration   : t별 구간 오차 평균·표준편차 (evaluate.py calibration.json과 같은 값)
#     maha              : Mahalanobis용 평균·공분산 (evaluate.py와 같은 값)
#     frame_calibration : t별 프레임 오차 평균·표준편차 → 시간축 설명(z 곡선)에 쓴다.
#     dev_scores        : dev real 파일 점수 세 가지 (백분위, 분포 그림용)
#     thresholds        : dev real 95백분위 임계값 (evaluate.py가 출력한 값과 같아야 한다)
#   last_result.json : 파일 하나를 분석한 마지막 결과 (두 번째 사용법)
# ===============================

import json  # reference.json 읽고 쓰기
import os  # 파일 경로
import sys  # 명령줄 인자
import time  # 기준값을 만든 시각

import librosa  # 화면용 Mel 그림 (모델 입력과 무관)
import numpy as np  # 설명 계산
import torch  # 모델 실행

from config import *  # config.py의 모든 설정값
from evaluate import (SCORE_KEYS, TIME_COLUMNS, calibrate, fit_real_distribution,  # 점수 계산을 evaluate.py와 똑같이 한다.
                      load_stats, make_fixed_noises, time_errors)
from preprocess import cut_segments, load_audio, make_silence_mask, normalize_peak, pad_to_segment, read_protocol, trim_silence, wave_to_input  # 전처리를 학습 때와 똑같이 한다.
from train import forward_diffusion, make_model  # 모델과 노이즈 섞기
from feature_maha import distance, load_reference, pooled_feature  # 특징 거리 점수 (판정용)

NOTE_Z = 1.0  # t별 평균 z가 이보다 크거나(-이보다 작으면) 요약 문장에서 "차이가 있다"고 말한다. (실험하면서 조정)
KEY_NAMES = {"score_mean": "평균 이상도", "score_max": "최대 구간 이상도", "score_maha": "분포 거리", "score_feature": "특징 거리"}  # 화면·문장에 쓰는 점수 이름
WEB_KEYS = SCORE_KEYS + (["score_feature"] if FEATURE_MAHA else [])  # 웹에 내는 점수: 디퓨전 점수 3가지 + 특징 거리(feature_maha.py)
MIN_AUDIO_SEC = 0.5  # 이보다 짧으면 분석하지 않는다. (WavLM은 최소 400샘플이 필요하고, 너무 짧으면 의미가 없다)
LOSSY_CODECS = {"mp3", "aac", "opus", "vorbis", "amr_nb", "amr_wb", "wmav1", "wmav2", "ac3", "gsm", "gsm_ms"}  # 손실 압축 (ffmpeg 이름)
LOSSY_EXTS = {"mp3", "m4a", "aac", "mp4", "ogg", "oga", "opus", "webm", "amr", "wma", "3gp"}  # 코덱을 모를 때 확장자로 판단

FRAME_SEC = FRAME_HOP / SAMPLE_RATE  # 모델 프레임 간격(초): WavLM 0.02
FRAME_OFFSET_SEC = (WIN_LENGTH / 2 / SAMPLE_RATE) if FEATURE_TYPE == "wavlm" else 0.0  # WavLM 프레임 i는 [320i, 320i+400) 샘플을 본다 → 중심이 12.5ms 뒤
DATA_DESC = {  # 화면에 보여 줄 데이터 이름
    "2019_train": "ASVspoof 2019 영어(스튜디오)",
    "ko_train": "AIHub 다화자 음성합성 한국어(4명)",
    "kspon_train": "AIHub 한국어 음성(KsponSpeech, 대화 세션 124개)",
}
if FEATURE_MAHA:  # 판정에 쓰는 진짜 음성 무리 설명 (특징 거리)
    TRAIN_DESC = " + ".join(DATA_DESC.get(d, d) for d in FEATURE_MAHA_TRAIN) + (", 방 환경" if FEATURE_MAHA_ROOM else "") + (", 목소리 높이 증강" if FEATURE_MAHA_SPEEDS else "")
else:
    TRAIN_DESC = "ASVspoof 2019 LA 진짜 음성, 스튜디오 녹음" + (", 전화 코덱 증강 포함" if AUGMENT_CODEC else "") + (", 방 환경 증강 포함" if AUGMENT_ROOM else "")  # 경고 문장에 쓰는 학습 데이터 설명


# ===============================
# 1. 준비
# ===============================

def load_model_with_info():  # best.pt를 불러오고, 몇 번째 epoch의 모델인지도 돌려준다.

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best.pt"), map_location=DEVICE)
    model = make_model().to(DEVICE)  # FEATURE_TYPE에 맞는 U-Net (WavLM이면 1D)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, {"epoch": int(checkpoint["epoch"]), "val_loss": float(checkpoint["val_loss"])}


# ===============================
# 2. 구간 자르기와 프레임별 오차
# ===============================

def prepare_segments(features, mask):  # 정규화된 특징 → 2초 창·1초 hop 구간 (evaluate.py measure_file과 같은 방식)

    segments = cut_segments(features, mask, hop_frames=INFER_HOP_FRAMES)

    if len(segments) == 0:  # 1초보다 짧은 파일
        piece, piece_mask = pad_to_segment(features, mask)  # 파일 전체를 한 조각으로 채워서 쓴다.
        segments = [(piece, piece_mask, 0)]

    speech_segments = [s for s in segments if s[1].sum() > 0]  # mask가 전부 0인(무음) 구간은 뺀다. (evaluate.py와 같다)
    if len(speech_segments) > 0:
        segments = speech_segments

    x = np.stack([s[0] for s in segments])[:, None]  # (S, 1, N_BINS, F)
    masks = np.stack([s[1] for s in segments])  # (S, F)
    starts = [s[2] for s in segments]  # 구간 시작 프레임

    return x, masks, starts


@torch.no_grad()  # gradient를 계산하지 않는다.
def frame_band_errors(model, x, fixed_noises):  # 구간 → 프레임마다·대역마다 (예측 노이즈 − 정답)²의 평균 (S, t 3, 대역 수, F)
    # evaluate.py band_mse가 프레임 방향으로 평균 내기 직전의 값이다. 프레임 방향으로 남겨 두면 "언제" 이상했는지 보인다.

    out = []
    bins_per_band = N_BINS // NUM_BANDS  # WavLM이면 대역 1개 = 768칸 전체

    for begin in range(0, x.shape[0], SCORE_CHUNK):  # 긴 파일은 SCORE_CHUNK개 구간씩 (GPU 메모리)
        x0 = torch.from_numpy(x[begin:begin + SCORE_CHUNK]).to(DEVICE)  # (n, 1, N_BINS, F)
        n = x0.shape[0]

        x_list, t_list, noise_list = [], [], []  # t 3개 × 노이즈 2개를 한 배치로 (evaluate.py와 같은 순서)
        for t_value in SCORE_TIMESTEPS:
            for noise in fixed_noises:
                t = torch.full((n,), t_value, device=DEVICE)
                noise_n = noise.expand(n, -1, -1, -1)
                x_t, _ = forward_diffusion(x0, t, noise_n)
                x_list.append(x_t)
                t_list.append(t)
                noise_list.append(noise_n)

        pred_noise = model(torch.cat(x_list), torch.cat(t_list))  # (6n, 1, N_BINS, F)
        squared = (pred_noise - torch.cat(noise_list)) ** 2
        squared = squared.view(len(SCORE_TIMESTEPS), len(fixed_noises), n, NUM_BANDS, bins_per_band, SEGMENT_FRAMES)
        squared = squared.mean(dim=(1, 4))  # 노이즈 2개 평균, 대역 안의 칸 평균 → (3, n, 대역, F)
        out.append(squared.permute(1, 0, 2, 3).cpu().numpy())  # (n, 3, 대역, F)
        # GroupNorm은 조각마다 따로 정규화하므로, 묶음을 어떻게 나눠도 결과는 같다.

    return np.concatenate(out)  # (S, 3, 대역, F)


def segment_errors(frame_errors, masks):  # (S, 3, 대역, F) → (S, 3 × 대역): evaluate.py band_mse와 같은 값, ERROR_COLUMNS 순서

    weights = masks[:, None, None, :].astype(np.float64)  # (S, 1, 1, F): mask=1인 프레임만
    total = (frame_errors * weights).sum(axis=3)  # (S, 3, 대역)
    count = np.maximum(masks.sum(axis=1), LOSS_EPS)[:, None, None]  # (S, 1, 1)

    return (total / count).reshape(len(frame_errors), -1)  # t 바깥, 대역 안쪽


def segment_z(time_errs, seg_calibration):  # (S, 3) → t별 z (S, 3). 구간 점수 = 평균 (evaluate.py segment_scores와 같다)

    z = np.zeros_like(time_errs)
    for j, column in enumerate(TIME_COLUMNS):
        z[:, j] = (time_errs[:, j] - seg_calibration[column]["mean"]) / seg_calibration[column]["std"]

    return z


def file_feature(errors):  # 파일의 구간 오차 (S, 24 또는 3) → Mahalanobis 특징 (evaluate.py file_features와 같다)
    return np.log(np.maximum(errors.mean(axis=0), LOSS_EPS))


def overlap_frames(frame_errs, masks, starts, num_frames):  # 1초씩 겹친 구간의 프레임 오차를 평균해 파일 길이 (3, N)로

    total = np.zeros((frame_errs.shape[1], num_frames), dtype=np.float64)
    count = np.zeros(num_frames, dtype=np.float64)

    for errs, mask, start in zip(frame_errs, masks, starts):  # 구간마다 (errs: (3, F))
        length = min(SEGMENT_FRAMES, num_frames - start)  # 오른쪽 패딩은 뺀다.
        valid = mask[:length]  # 무음 프레임도 뺀다.
        total[:, start:start + length] += errs[:, :length] * valid
        count[start:start + length] += valid

    covered = count > 0  # 한 번이라도 계산된 프레임 (무음은 False)
    out = np.zeros_like(total)
    out[:, covered] = total[:, covered] / count[covered]

    return out, covered


def merge_intervals(intervals):  # 겹치는 [시작, 끝]을 하나로 합친다.

    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def feature_labels():  # Mahalanobis 특징 이름 (WavLM이면 t 3개, 대역이 있으면 t × 대역)

    labels = []
    for t in SCORE_TIMESTEPS:
        for b in range(NUM_BANDS):
            labels.append(f"t={t}" if NUM_BANDS == 1 else f"t={t} · 대역 {b + 1}")

    return labels


# ===============================
# 3. 기준값 만들기 (dev real만)
# ===============================

def build_reference():

    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    model, checkpoint_info = load_model_with_info()
    stats = load_stats()
    fixed_noises = make_fixed_noises()
    print(f"모델: {RUN_NAME} best.pt (epoch {checkpoint_info['epoch']}), 입력: {FEATURE_TYPE}, 장치: {DEVICE}")

    items = [item for item in read_protocol(DEV_DATA) if item["label"] == "bonafide"]  # evaluate.py와 같은 dev real 전체
    print(f"[{DEV_DATA}] real 파일 {len(items)}개로 기준값 계산 시작")

    all_errors = []  # 파일마다 (구간 수, 3 × 대역)
    frame_sum = np.zeros(len(SCORE_TIMESTEPS), dtype=np.float64)  # t별 프레임 오차 합
    frame_sq_sum = np.zeros_like(frame_sum)
    frame_count = 0

    for i, item in enumerate(items):
        features, mask = wave_to_input(load_audio(item["path"]), stats)  # 정규화된 특징
        x, masks, starts = prepare_segments(features, mask)
        errors = frame_band_errors(model, x, fixed_noises)  # (S, 3, 대역, F)
        all_errors.append(segment_errors(errors, masks))

        frames, covered = overlap_frames(errors.mean(axis=2), masks, starts, features.shape[1])  # 화면에 보여 줄 때와 같은 방식
        used = frames[:, covered]  # (3, 사용한 프레임 수)
        frame_sum += used.sum(axis=1)
        frame_sq_sum += (used ** 2).sum(axis=1)
        frame_count += used.shape[1]

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)}")

    # ----- 점수 보정 (evaluate.py와 같은 계산) -----
    seg_calibration = calibrate(time_errors(np.concatenate(all_errors)))
    dev_features = {i: file_feature(e) for i, e in enumerate(all_errors)}
    maha_mean, maha_cov = fit_real_distribution(dev_features)
    maha_cov = np.atleast_2d(maha_cov)
    cov_inv = np.linalg.pinv(maha_cov)

    dev_scores = {key: [] for key in SCORE_KEYS}
    for i, errors in enumerate(all_errors):
        seg_scores = segment_z(time_errors(errors), seg_calibration).mean(axis=1)
        diff = dev_features[i] - maha_mean
        dev_scores["score_mean"].append(float(seg_scores.mean()))
        dev_scores["score_max"].append(float(seg_scores.max()))
        dev_scores["score_maha"].append(float(np.sqrt(max(diff @ cov_inv @ diff, 0.0))))

    thresholds = {key: float(np.percentile(values, THRESHOLD_PERCENTILE)) for key, values in dev_scores.items()}

    frame_mean = frame_sum / frame_count
    frame_std = np.maximum(np.sqrt(np.maximum(frame_sq_sum / frame_count - frame_mean ** 2, 0.0)), 1e-8)

    reference = {
        "meta": {  # 기준값을 만들 때의 설정. 서버를 켤 때 지금 설정·모델과 같은지 확인한다.
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": RUN_NAME,
            "checkpoint": checkpoint_info,
            "feature_type": FEATURE_TYPE,
            "wavlm": [WAVLM_MODEL_NAME, WAVLM_LAYER] if FEATURE_TYPE == "wavlm" else None,
            "score_timesteps": SCORE_TIMESTEPS,
            "noise_seeds": SCORE_NOISE_SEEDS,
            "num_bands": NUM_BANDS,
            "mask_silence": MASK_SILENCE,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "dev_data": DEV_DATA,
            "num_dev_files": len(items),
        },
        "seg_calibration": seg_calibration,
        "maha": {"mean": maha_mean.tolist(), "cov": maha_cov.tolist()},
        "frame_calibration": {"mean": frame_mean.tolist(), "std": frame_std.tolist()},
        "dev_scores": dev_scores,
        "thresholds": thresholds,
    }
    with open(REFERENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(reference, f)

    # ----- evaluate.py 결과와 비교 (있으면) -----
    metrics_path = os.path.join(EVAL_DIR, "metrics.json")
    saved = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    for key in SCORE_KEYS:
        line = f"임계값 {key}: {thresholds[key]:.4f}"
        if saved and key in saved:
            gap = abs(saved[key]["threshold"] - thresholds[key])
            line += f"  (evaluate.py {saved[key]['threshold']:.4f}, 차이 {gap:.1e} → {'같음' if gap < 1e-3 else '다름! 설정이나 best.pt를 확인'})"
        print(line)
    print(f"저장: {REFERENCE_PATH}")


# ===============================
# 4. 입력 점검 (학습 데이터와 얼마나 다른 조건인지)
# ===============================

def inspect_audio(wave, source_info, original_sec, truncated):  # real-only 모델은 낯선 조건을 모두 "이상"으로 볼 수 있어서 따로 알린다.

    source_info = source_info or {}
    notes = []  # {"level": "warn" 또는 "info", "text": 문장}

    duration = len(wave) / SAMPLE_RATE
    peak = float(np.abs(wave).max())
    clip_ratio = float((np.abs(wave) >= 0.999).mean())
    silence_ratio = float(1.0 - make_silence_mask(wave, len(wave)).mean())  # 무음 프레임 비율 (MASK_SILENCE와 상관없이 잰다)

    sample_rate = source_info.get("sample_rate")
    channels = source_info.get("channels")
    codec = (source_info.get("codec") or "").lower()
    ext = (source_info.get("ext") or "").lower()
    is_lossy = codec in LOSSY_CODECS or (not codec and ext in LOSSY_EXTS)

    if source_info.get("source") == "mic":
        notes.append({"level": "warn", "text": f"마이크로 녹음한 음성입니다. 학습 데이터({TRAIN_DESC})와 녹음 환경이 달라 진짜 음성도 점수가 높게 나올 수 있습니다."})
    if truncated:
        notes.append({"level": "info", "text": f"{original_sec:.1f}초 중 앞 {MAX_AUDIO_SEC:g}초만 분석했습니다."})
    if duration < MIN_SEGMENT_SEC:
        notes.append({"level": "warn", "text": f"{duration:.2f}초로 1초보다 짧아 0으로 채워서 분석했습니다. 결과가 불안정할 수 있습니다."})
    if sample_rate and sample_rate < SAMPLE_RATE:
        notes.append({"level": "warn", "text": f"원래 샘플레이트가 {sample_rate} Hz입니다. {sample_rate / 2000:g} kHz 이상 정보가 비어 있어 학습 데이터(16 kHz)와 다르고, 진짜 음성도 점수가 높게 나올 수 있습니다."})
    elif sample_rate and sample_rate > SAMPLE_RATE:
        notes.append({"level": "info", "text": f"{sample_rate} Hz를 16 kHz로 낮춰 분석했습니다."})
    if is_lossy:
        notes.append({"level": "warn", "text": f"손실 압축 형식({codec or ext})입니다. 학습 데이터(FLAC, 무손실)와 달라 진짜 음성도 점수가 높게 나올 수 있습니다."})
    if channels and channels > 1:
        notes.append({"level": "info", "text": f"{channels}채널을 모노로 합쳐서 분석했습니다."})
    if clip_ratio > 0.001:
        notes.append({"level": "warn", "text": f"소리가 깨진(클리핑) 샘플이 {clip_ratio * 100:.2f}% 있습니다."})
    if peak < 0.05:
        notes.append({"level": "warn", "text": f"소리가 매우 작습니다. (최대 진폭 {peak:.3f}) 크기는 맞춰서 분석하지만, 배경 잡음도 같이 커지므로 마이크에 더 가까이 대고 녹음하는 편이 좋습니다."})
    if silence_ratio > 0.5:
        notes.append({"level": "warn", "text": f"무음 비율이 {silence_ratio * 100:.0f}%로 높아 분석에 쓸 수 있는 구간이 적습니다."})

    quality = {
        "duration": duration,
        "peak": peak,
        "clip_ratio": clip_ratio,
        "silence_ratio": silence_ratio,
        "sample_rate": sample_rate,
        "channels": channels,
        "format": codec or ext or None,
    }

    return quality, notes


# ===============================
# 5. 판정과 설명
# ===============================

def make_summary(r):  # 숫자로부터 규칙에 따라 문장을 만든다. (근거 없는 말이 섞이지 않도록 생성형 모델은 쓰지 않는다)

    lines = []
    key, name = r["decision_key"], KEY_NAMES[r["decision_key"]]

    if r["verdict"] == "spoof":
        lines.append(f"판정 기준인 {name}가 임계값보다 높아({r['score']:.2f} > {r['threshold']:.2f}) 진짜 음성과 다른 패턴으로 판단했습니다.")
    else:
        lines.append(f"판정 기준인 {name}가 임계값 이하라서({r['score']:.2f} ≤ {r['threshold']:.2f}) 진짜 음성 범위 안에 있습니다.")

    flagged = [KEY_NAMES[k] for k in WEB_KEYS if r["flags"][k]]
    if len(flagged) == len(WEB_KEYS):
        lines.append(f"점수 {len(WEB_KEYS)}가지 모두 임계값을 넘었습니다.")
    elif len(flagged) == 0:
        lines.append(f"점수 {len(WEB_KEYS)}가지 모두 임계값 이하입니다.")
    else:
        lines.append(f"점수 {len(WEB_KEYS)}가지 중 {len(flagged)}가지가 임계값을 넘었습니다({', '.join(flagged)}).")

    lines.append(f"개발용 진짜 음성 {r['reference']['num_dev_files']}개와 비교하면 {r['percentile']:.1f}%가 이 파일보다 {name}가 낮습니다.")

    top = max(r["segments"], key=lambda s: s["score"])
    if r["flagged_intervals"]:
        spans = ", ".join(f"{a:.1f}–{b:.1f}초" for a, b in r["flagged_intervals"])
        lines.append(f"구간 기준({r['segment_threshold']:.2f})을 넘은 구간: {spans}. 가장 높은 구간은 {top['start']:.1f}–{top['end']:.1f}초(구간 점수 {top['score']:.2f})입니다.")
    else:
        lines.append(f"구간 기준({r['segment_threshold']:.2f})을 넘은 구간은 없습니다. 가장 높은 구간은 {top['start']:.1f}–{top['end']:.1f}초(구간 점수 {top['score']:.2f})입니다.")

    t_mean = r["t_contrib"]["mean"]
    j = int(np.argmax(np.abs(t_mean)))
    t = SCORE_TIMESTEPS[j]
    if t_mean[j] >= NOTE_Z:
        lines.append(f"노이즈 단계로 보면 t={t}에서 오차가 진짜 음성보다 가장 크게 벌어졌습니다(평균 z {t_mean[j]:+.2f}).")
    elif t_mean[j] <= -NOTE_Z:
        lines.append(f"노이즈 단계로 보면 t={t}에서 오차가 진짜 음성보다 오히려 작았습니다(평균 z {t_mean[j]:+.2f}). 일부 합성 음성은 지나치게 매끄러워 오차가 작게 나오며, 분포 거리 점수가 이런 경우를 잡습니다.")
    else:
        lines.append("노이즈 단계(t)별 오차는 진짜 음성 평균과 크게 다르지 않습니다.")

    features = r["maha"]["features"]
    if r["flags"]["score_maha"] and features:  # 거리가 임계값을 넘었을 때만 말한다. (거리가 작으면 기여율이 들쭉날쭉해 의미가 없다)
        top_f = max(features, key=lambda f: f["share"])
        direction = "큰" if top_f["diff_sd"] > 0 else "작은"
        lines.append(f"분포 거리에는 {top_f['label']} 특징이 가장 크게 기여했습니다(기여 {top_f['share'] * 100:.0f}%, 진짜 음성보다 오차가 {direction} 방향 {top_f['diff_sd']:+.1f}σ).")

    num_warn = sum(note["level"] == "warn" for note in r["notes"])
    if num_warn > 0:
        lines.append(f"입력 조건 경고가 {num_warn}개 있습니다. 학습 데이터와 조건이 다르면 진짜 음성도 점수가 높아질 수 있으니 '입력 점검'을 함께 보세요.")

    return lines


class Explainer:  # 서버를 켤 때 한 번 만들고, 요청마다 analyze_wave()를 부른다.

    def __init__(self):

        if not os.path.exists(REFERENCE_PATH):
            raise FileNotFoundError(f"기준값 파일이 없습니다: {REFERENCE_PATH}\n먼저 python src/explain.py 를 실행하세요.")
        with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
            self.reference = json.load(f)

        self.model, self.checkpoint = load_model_with_info()
        self.stats = load_stats()
        self.fixed_noises = make_fixed_noises()
        self.check_reference()  # 기준값과 지금 모델·설정이 짝이 맞는지

        ref = self.reference
        self.seg_calibration = ref["seg_calibration"]
        self.thresholds = ref["thresholds"]
        self.maha_mean = np.array(ref["maha"]["mean"])
        self.maha_cov = np.atleast_2d(np.array(ref["maha"]["cov"]))
        self.maha_cov_inv = np.linalg.pinv(self.maha_cov)
        self.frame_mean = np.array(ref["frame_calibration"]["mean"])[:, None]  # (3, 1)
        self.frame_std = np.array(ref["frame_calibration"]["std"])[:, None]
        self.dev_scores = {key: np.array(values) for key, values in ref["dev_scores"].items()}

        if FEATURE_MAHA:  # 특징 거리 기준값 (python src/feature_maha.py 가 만든다)
            if not os.path.exists(FEATURE_REFERENCE_PATH):
                raise FileNotFoundError(f"특징 거리 기준값이 없습니다: {FEATURE_REFERENCE_PATH}\n먼저 python src/feature_maha.py 를 실행하세요.")
            self.feature_ref = load_reference()
            meta = str(self.feature_ref["meta"])
            if not meta.startswith(f"{WAVLM_MODEL_NAME}|{WAVLM_LAYER}|"):  # 다른 모델·층으로 만든 기준값이면 점수가 조용히 틀어진다.
                raise RuntimeError(f"특징 거리 기준값의 모델·층이 지금 설정과 다릅니다: {meta}\npython src/feature_maha.py 를 다시 실행하세요.")
            self.thresholds["score_feature"] = float(np.percentile(self.feature_ref["dev_scores"], FEATURE_THRESHOLD_PERCENTILE))  # dev real 점수에서 바로 계산 (백분위를 바꿔도 기준값을 다시 만들 필요 없음)
            self.dev_scores["score_feature"] = self.feature_ref["dev_scores"]

        self.hists = {}
        for key, values in self.dev_scores.items():  # 화면의 분포 그림용
            counts, edges = np.histogram(values, bins=40)
            self.hists[key] = {"counts": counts.tolist(), "edges": edges.tolist()}

        self.display_mel_hz = librosa.mel_frequencies(n_mels=DISPLAY_N_MELS + 2, fmin=0.0, fmax=SAMPLE_RATE / 2)[1:-1]  # 화면용 Mel 칸의 중심 주파수

    def check_reference(self):  # 설정이나 best.pt가 바뀌었는데 예전 기준값을 쓰면 점수가 조용히 틀어진다. 그래서 멈춘다.

        meta = self.reference["meta"]
        now = {
            "run_name": RUN_NAME, "feature_type": FEATURE_TYPE,
            "wavlm": [WAVLM_MODEL_NAME, WAVLM_LAYER] if FEATURE_TYPE == "wavlm" else None,
            "score_timesteps": SCORE_TIMESTEPS, "noise_seeds": SCORE_NOISE_SEEDS, "num_bands": NUM_BANDS,
            "mask_silence": MASK_SILENCE, "threshold_percentile": THRESHOLD_PERCENTILE, "dev_data": DEV_DATA,
        }
        problems = [f"{name}: {meta.get(name)} → {value}" for name, value in now.items() if meta.get(name) != value]
        saved = meta["checkpoint"]
        if saved["epoch"] != self.checkpoint["epoch"] or abs(saved["val_loss"] - self.checkpoint["val_loss"]) > 1e-9:
            problems.append(f"best.pt: epoch {saved['epoch']} → {self.checkpoint['epoch']}")

        if problems:
            raise RuntimeError("기준값을 만들 때와 지금이 다릅니다. " + ", ".join(problems) + "\npython src/explain.py 를 다시 실행하세요.")

    def model_card(self):  # 화면과 보고서에 적을 모델 정보

        feature = {
            "wavlm": f"{WAVLM_MODEL_NAME} {WAVLM_LAYER}층 특징",
            "mel": f"log-Mel {N_MELS}칸",
            "spec": f"log 스펙트로그램 {N_BINS}칸",
        }[FEATURE_TYPE]

        return {
            "run_name": RUN_NAME,
            "feature_type": FEATURE_TYPE,
            "feature": feature,
            "network": "1D U-Net 디퓨전(DDPM)" if FEATURE_TYPE == "wavlm" else "2D U-Net 디퓨전(DDPM)",
            "decision_method": "특징 거리(마할라노비스, 디퓨전 없음)" if DECISION_KEY == "score_feature" else "디퓨전 노이즈 예측 오차",  # 무엇으로 판정했는지
            "room_augment": AUGMENT_ROOM,
            "feature_train_desc": TRAIN_DESC if FEATURE_MAHA else None,  # 특징 거리의 진짜 음성 무리 (화면 "방법"에 표시)
            "feature_threshold_percentile": FEATURE_THRESHOLD_PERCENTILE if FEATURE_MAHA else None,
            "feature_num_real": int(str(self.feature_ref["meta"]).split("|train=")[1].split(":")[1].split("|")[0]) if FEATURE_MAHA else None,  # 무리에 들어간 진짜 음성 수
            "checkpoint_epoch": self.checkpoint["epoch"],
            "train_data": TRAIN_DATA,
            "dev_data": DEV_DATA,
            "codec_augment": AUGMENT_CODEC,
            "score_timesteps": SCORE_TIMESTEPS,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "decision_key": DECISION_KEY,
            "num_dev_files": self.reference["meta"]["num_dev_files"],
            "reference_created": self.reference["meta"]["created"],
        }

    def info(self):  # GET /api/info
        return {"model": self.model_card(), "thresholds": self.thresholds, "max_audio_sec": MAX_AUDIO_SEC, "max_upload_mb": MAX_UPLOAD_MB}

    def analyze(self, path, source_info=None):  # 파일 경로 → 결과 (명령줄 확인용)
        return self.analyze_wave(load_audio(path), source_info)

    def analyze_wave(self, wave, source_info=None):  # 16kHz 모노 waveform → 판정과 설명 (dict)

        original_sec = len(wave) / SAMPLE_RATE
        if float(np.abs(wave).max()) == 0.0:
            raise ValueError("소리가 없는 파일입니다.")

        trim_offset = 0.0  # 앞에서 잘라 낸 무음 길이(초). 웹에서 재생 위치를 맞출 때 쓴다.
        if TRIM_SILENCE:  # 학습·평가 때처럼 앞뒤 무음을 잘라 낸다. (load_audio와 같은 함수)
            wave, start = trim_silence(wave)
            trim_offset = start / SAMPLE_RATE

        max_samples = int(MAX_AUDIO_SEC * SAMPLE_RATE)
        truncated = len(wave) > max_samples
        if truncated:
            wave = wave[:max_samples]  # 앞부분만 분석한다.

        if len(wave) < MIN_AUDIO_SEC * SAMPLE_RATE:
            raise ValueError(f"음성이 너무 짧습니다. 앞뒤 무음을 빼고 {MIN_AUDIO_SEC:g}초 이상이어야 합니다.")

        quality, notes = inspect_audio(wave, source_info, original_sec, truncated)
        if TRIM_SILENCE:
            notes.append({"level": "info", "text": f"학습 때처럼 앞뒤 무음을 잘라 내고 {len(wave) / SAMPLE_RATE:.1f}초만 분석했습니다. (원래 {original_sec:.1f}초)"})
        if NORMALIZE_PEAK:  # 학습 때처럼 소리 크기를 맞춘다. (quality의 최대 진폭은 맞추기 전 원래 값)
            notes.append({"level": "info", "text": f"학습 때처럼 소리 크기를 맞춰(최대 진폭 {quality['peak']:.2f} → 1.0) 분석했습니다."})
            wave = normalize_peak(wave)

        features, mask = wave_to_input(wave, self.stats)  # 학습 때와 같은 특징·mask·정규화
        if mask.sum() == 0:
            raise ValueError("말소리가 있는 구간을 찾지 못했습니다. (모든 프레임이 무음으로 판단됨)")

        x, masks, starts = prepare_segments(features, mask)
        errors = frame_band_errors(self.model, x, self.fixed_noises)  # (S, 3, 대역, F)

        feature_score = None
        if FEATURE_MAHA:  # 판정용 특징 거리. 음성 모델을 한 번 더 돌리지만(위 wave_to_input에서 한 번), 웹에서는 파일 하나라 빠르다.
            feature_score = float(distance(pooled_feature(wave), self.feature_ref)[0])

        return self.build_result(wave, features.shape[1], errors, masks, starts, quality, notes, original_sec, trim_offset, feature_score)

    def build_result(self, wave, num_frames, errors, masks, starts, quality, notes, original_sec, trim_offset=0.0, feature_score=None):  # 오차 → 점수·설명 (모델 없이 numpy만)

        duration = quality["duration"]

        # ----- 점수 세 가지 (evaluate.py와 같은 계산) -----
        seg_errors = segment_errors(errors, masks)  # (S, 3 × 대역)
        z_t = segment_z(time_errors(seg_errors), self.seg_calibration)  # (S, 3)
        seg_scores = z_t.mean(axis=1)

        feature = file_feature(seg_errors)
        diff = feature - self.maha_mean
        weighted = self.maha_cov_inv @ diff
        maha_sq = max(float(diff @ weighted), 0.0)

        scores = {
            "score_mean": float(seg_scores.mean()),
            "score_max": float(seg_scores.max()),
            "score_maha": float(np.sqrt(maha_sq)),
        }
        if feature_score is not None:
            scores["score_feature"] = feature_score  # 특징 거리 (디퓨전을 쓰지 않는 판정용 점수)
        flags = {key: bool(scores[key] > self.thresholds[key]) for key in WEB_KEYS}
        percentiles = {key: float((self.dev_scores[key] < scores[key]).mean() * 100) for key in WEB_KEYS}

        # ----- 설명 1: 시간 (구간 점수, 프레임 z) -----
        segment_threshold = self.thresholds["score_max"]  # dev real 파일의 "가장 높은 구간"의 95백분위 = 구간 하나가 낯선지 보는 기준
        segments = []
        for start, score, z in zip(starts, seg_scores, z_t):
            start_sec = start * FRAME_SEC
            segments.append({
                "start": round(start_sec, 3),
                "end": round(min(start_sec + SEGMENT_SEC, duration), 3),
                "score": float(score),
                "z_t": [float(v) for v in z],
                "flagged": bool(score > segment_threshold),
            })
        flagged_intervals = merge_intervals([[s["start"], s["end"]] for s in segments if s["flagged"]])

        frames, covered = overlap_frames(errors.mean(axis=2), masks, starts, num_frames)  # (3, N)
        frame_z = (frames - self.frame_mean) / self.frame_std  # t별로 dev real 프레임 기준 z
        frame_z[:, ~covered] = 0.0

        # ----- 설명 3: 분포 거리 기여 -----
        contributions = diff * weighted  # 합하면 거리²가 된다. (특징끼리 상관이 있으면 음수도 나올 수 있다)
        sds = np.sqrt(np.maximum(np.diag(self.maha_cov), 1e-12))
        maha_features = []
        for label, d, c, sd in zip(feature_labels(), diff, contributions, sds):
            maha_features.append({
                "label": label,
                "diff_sd": float(d / sd),  # real 평균에서 몇 σ 떨어졌는지 (+: 오차가 큼, −: 오차가 작음)
                "contribution": float(c),
                "share": float(c / maha_sq) if maha_sq > 0 else 0.0,  # 거리²에서 차지하는 비율
            })

        # ----- 화면용 Mel (모델 입력 아님) -----
        mel = librosa.feature.melspectrogram(y=wave, sr=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
                                             hop_length=FRAME_HOP, n_mels=DISPLAY_N_MELS, power=2.0)
        mel_db = np.round(10 * np.log10(mel + 1e-10)).astype(int)  # 정수 dB (전송량을 줄이고 세부를 덜 담는다)

        top = int(np.argmax(seg_scores))
        result = {
            "verdict": "spoof" if flags[DECISION_KEY] else "bonafide",
            "decision_key": DECISION_KEY,
            "score": scores[DECISION_KEY],
            "threshold": self.thresholds[DECISION_KEY],
            "percentile": percentiles[DECISION_KEY],
            "scores": scores,
            "thresholds": self.thresholds,
            "percentiles": percentiles,
            "flags": flags,
            "duration": duration,
            "original_duration": original_sec,
            "trim_offset": trim_offset,  # 분석 시간 0초 = 원래 녹음의 trim_offset초 (웹 재생 위치 맞추기)
            "segment_sec": SEGMENT_SEC,
            "segment_threshold": segment_threshold,
            "segments": segments,
            "flagged_intervals": flagged_intervals,
            "frames": {
                "frame_sec": FRAME_SEC,
                "offset_sec": FRAME_OFFSET_SEC,
                "num_frames": int(num_frames),
                "z_t": np.round(frame_z, 2).tolist(),  # (3, N)
                "covered": covered.astype(int).tolist(),
            },
            "t_contrib": {
                "timesteps": SCORE_TIMESTEPS,
                "mean": [float(v) for v in z_t.mean(axis=0)],  # 파일 전체 평균 z
                "top_segment": [float(v) for v in z_t[top]],  # 가장 높은 구간의 z
            },
            "maha": {"distance": scores["score_maha"], "features": maha_features},
            "display": {
                "frame_sec": FRAME_HOP / SAMPLE_RATE,
                "mel_hz": [round(float(v)) for v in self.display_mel_hz],
                "mel_db": mel_db.tolist(),  # (DISPLAY_N_MELS, M), 0번 줄이 가장 낮은 주파수
            },
            "reference": {"num_dev_files": self.reference["meta"]["num_dev_files"], "hists": self.hists},
            "model": self.model_card(),
            "quality": quality,
            "notes": notes,
        }
        result["summary"] = make_summary(result)

        return result


# ===============================
# 6. 실행
# ===============================

if __name__ == "__main__":

    if len(sys.argv) == 1:  # python src/explain.py
        build_reference()

    else:  # python src/explain.py 음성파일경로
        explainer = Explainer()
        result = explainer.analyze(sys.argv[1], {"ext": os.path.splitext(sys.argv[1])[1].lstrip(".")})

        print(f"판정: {'딥보이스 의심' if result['verdict'] == 'spoof' else '진짜 음성 범위'} (기준 {DECISION_KEY})")
        for key in SCORE_KEYS:
            print(f"  {key}: {result['scores'][key]:.4f} / 임계값 {result['thresholds'][key]:.4f}")
        for line in result["summary"]:
            print(f"  - {line}")
        for note in result["notes"]:
            print(f"  [{note['level']}] {note['text']}")

        os.makedirs(EXPLAIN_DIR, exist_ok=True)
        with open(os.path.join(EXPLAIN_DIR, "last_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        print(f"전체 결과: {os.path.join(EXPLAIN_DIR, 'last_result.json')}")
