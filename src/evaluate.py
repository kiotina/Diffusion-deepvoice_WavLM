# ===============================
# evaluate.py
# 학습한 DDPM의 노이즈 예측 오차로 파일마다 이상 점수를 만들고, eval에서 EER·AUROC를 계산한다.
#
# 순서
#   1. 측정: dev real 전체와 eval 파일을 2초 창·1초 hop으로 자르고, 구간마다 (t, 채널)별 오차를 잰다.
#   2. 보정: dev real 구간의 (t, 채널)별 평균·표준편차로 오차를 z-점수로 바꾼다. (real-only)
#   3. 판정: 파일 점수(평균, 최댓값) → dev real 95백분위 임계값 → eval 지표
#   4. 시험지: TEST_SAMPLES_DIR의 음성마다 같은 방식으로 점수와 판정
#
# 파일 점수 두 가지 (모두 "클수록 fake")
#   score_mean : 구간 점수의 평균. 파일 전체가 얼마나 real답지 않은지
#   score_max  : 구간 점수의 최댓값. 가장 의심스러운 구간 하나로 판단한다.
#
# 만드는 파일 (outputs/<RUN_NAME>/evaluation_<EVAL_DATA> 폴더)
#   segments.csv       : 구간마다 (t, 채널)별 오차 (파일 단위로 바로 기록 → 중간에 멈춰도 이어서 실행)
#   calibration.json   : (t, 채널)별 평균·표준편차, 임계값 (dev real로 계산)
#   predictions.csv    : eval 파일마다 점수와 판정
#   metrics.json       : EER, AUROC, 임계값에서 오탐률·탐지율, 공격별 지표
#   score_hist.png     : real·fake 점수 분포와 임계값
#   test_samples.csv   : 시험지 파일마다 점수와 판정
# ===============================

import csv  # segments.csv, predictions.csv를 쓴다.
import json  # calibration.json, metrics.json을 쓴다.
import os  # 파일 경로를 다룬다.

import matplotlib  # 그림을 저장한다.
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt  # 점수 분포를 그린다.
import numpy as np  # 점수 계산
import torch  # 모델 실행
from sklearn.metrics import roc_auc_score, roc_curve  # AUROC, ROC 곡선(EER 계산용)

from config import *  # config.py의 모든 설정값을 가져온다.
from preprocess import cut_segments, file_to_feature, load_stats, pad_to_segment, read_protocol  # 전처리와 똑같이 자른다.
from train import UNet, forward_diffusion  # 모델과 노이즈 섞기 (학습과 똑같이)

# segments.csv에는 넓게 잰 오차를 모두 남긴다: (측정 t) × 채널 × 대역. 예: e_t10_spec_b0, …, e_t400_delta_b7
ERROR_COLUMNS = [f"e_t{t}_{name}_b{b}" for t in MEASURE_TIMESTEPS for name in FEATURES for b in range(NUM_BANDS)]
# 기본 점수(score_mean, score_max)는 SCORE_TIMESTEPS의 (t, 채널)별 오차만 쓴다. (대역 8개를 평균한 값 = 채널 전체 오차)
SCORE_COLUMNS = [f"e_t{t}_{name}" for t in SCORE_TIMESTEPS for name in FEATURES]
SEGMENT_COLUMNS = ["split", "file_id", "label", "attack_id", "start_sec"] + ERROR_COLUMNS  # segments.csv의 열
SCORE_KEYS = ["score_mean", "score_max"]  # 파일 점수 두 가지


# ===============================
# 1. 준비: 모델, 고정 노이즈
# ===============================

def load_model():  # best.pt(val loss가 가장 낮았던 모델)를 불러온다.

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best.pt"), map_location=DEVICE)
    if checkpoint["features"] != FEATURES:  # 학습 때와 특징 설정이 다르면 점수가 의미 없다.
        raise ValueError(f"모델은 {checkpoint['features']}로 학습했는데 config.py는 {FEATURES}이다.")

    model = UNet().to(DEVICE)
    model.load_state_dict(checkpoint["model"])  # 학습한 가중치를 넣는다.
    model.eval()  # 평가 모드
    print(f"모델: best.pt (epoch {checkpoint['epoch']}, val loss {checkpoint['val_loss']:.4f})")

    return model


def make_fixed_noises():  # seed마다 노이즈 하나를 만들어, 모든 파일·구간에 똑같이 쓴다.

    noises = []  # SCORE_NOISE_SEEDS 순서대로
    for seed in SCORE_NOISE_SEEDS:
        generator = torch.Generator().manual_seed(seed)  # 이 seed 전용 난수
        noise = torch.randn((1, N_CHANNELS, N_BINS, SEGMENT_FRAMES), generator=generator)  # (1, 채널 수, 256, 126)
        noises.append(noise.to(DEVICE))
    # 모든 파일이 같은 노이즈로 시험을 보므로, 점수 차이는 노이즈가 아니라 음성의 차이에서 나온다.

    return noises


# ===============================
# 2. 파일 하나의 구간별 오차 측정
# ===============================

def band_mse(pred, target, mask):  # 구간마다, 채널마다, 주파수 대역마다 masked MSE (train.py의 masked_mse를 나눠서 잰 것)

    batch = pred.shape[0]  # 구간 수 B
    bins_per_band = N_BINS // NUM_BANDS  # 대역 하나의 칸 수 (256 / 8 = 32칸 = 1kHz)

    mask = mask.expand_as(pred)  # (B, 1, 1, 126) → (B, 채널 수, 256, 126)
    squared_error = (pred - target) ** 2 * mask  # mask=0인 칸(패딩)의 오차는 0으로 지운다.

    squared_error = squared_error.reshape(batch, N_CHANNELS, NUM_BANDS, bins_per_band, SEGMENT_FRAMES)  # 256칸을 32칸씩 8묶음으로
    mask = mask.reshape(batch, N_CHANNELS, NUM_BANDS, bins_per_band, SEGMENT_FRAMES)

    error_sum = squared_error.sum(dim=(3, 4))  # (B, 채널 수, 대역 수): 대역마다 오차의 합
    count = mask.sum(dim=(3, 4))  # (B, 채널 수, 대역 수): 대역마다 mask=1인 칸 수

    return error_sum / count.clamp(min=LOSS_EPS)  # (B, 채널 수, 대역 수). 대역을 평균하면 채널 전체 오차와 같다.


@torch.no_grad()  # 평가에서는 gradient를 계산하지 않는다.
def measure_file(model, path, stats, fixed_noises):  # 음성 파일 → 구간마다 (t, 채널, 대역)별 오차 (구간 수, ERROR_COLUMNS 수)

    feature = file_to_feature(path, stats)  # 정규화된 특징 (채널 수, 256, 프레임 수)
    segments = cut_segments(feature, hop_frames=INFER_HOP_FRAMES)  # 2초 창을 1초씩 밀며 자른다.

    if len(segments) == 0:  # 1초보다 짧은 파일은 조각이 안 나온다.
        piece, mask = pad_to_segment(feature)  # 파일 전체를 한 조각으로 채워서 쓴다. (eval 파일을 빼먹지 않도록)
        segments = [(piece, mask, 0)]

    x0 = torch.from_numpy(np.stack([s[0] for s in segments])).to(DEVICE)  # (S, 채널 수, 256, 126)
    masks = torch.from_numpy(np.stack([s[1] for s in segments]))[:, None, None, :].to(DEVICE)  # (S, 1, 1, 126)
    starts = [s[2] for s in segments]  # 구간 시작 프레임
    num_segments = x0.shape[0]  # S

    # t 6개 × 노이즈 2개 = 12가지 경우를 한 배치로 묶어 U-Net을 한 번만 돌린다. (빠르게)
    x_list, t_list, noise_list = [], [], []
    for t_value in MEASURE_TIMESTEPS:  # t마다 (바깥 반복)
        for noise in fixed_noises:  # 노이즈마다 (안쪽 반복)
            t = torch.full((num_segments,), t_value, device=DEVICE)  # 모든 구간에 같은 t
            noise = noise.expand(num_segments, -1, -1, -1)  # 같은 노이즈를 구간 수만큼 늘린다.
            x_t, _ = forward_diffusion(x0, t, noise)  # 고정 노이즈를 섞는다.
            x_list.append(x_t)
            t_list.append(t)
            noise_list.append(noise)

    pred_noise = model(torch.cat(x_list), torch.cat(t_list))  # (12S, 채널 수, 256, 126)
    errors = band_mse(pred_noise, torch.cat(noise_list), masks.repeat(len(x_list), 1, 1, 1))  # (12S, 채널 수, 대역 수)
    errors = errors.view(len(MEASURE_TIMESTEPS), len(fixed_noises), num_segments, N_CHANNELS, NUM_BANDS)  # 위 반복 순서와 같다.
    errors = errors.mean(dim=1)  # 노이즈 2개의 평균 → (t 6, S, 채널 수, 대역 수)
    errors = errors.permute(1, 0, 2, 3).reshape(num_segments, -1)  # (S, t × 채널 × 대역): ERROR_COLUMNS 순서(t 바깥 → 채널 → 대역 안쪽)와 같다.

    return starts, errors.cpu().numpy()


def score_errors(errors):  # (구간 수, ERROR_COLUMNS 수) → (구간 수, SCORE_COLUMNS 수): 기본 점수용 t만 골라 대역을 평균한다.

    errors = errors.reshape(len(errors), len(MEASURE_TIMESTEPS), N_CHANNELS, NUM_BANDS)  # (S, t 6, 채널, 대역)
    picked = [MEASURE_TIMESTEPS.index(t) for t in SCORE_TIMESTEPS]  # 기본 점수에 쓸 t의 위치
    return errors[:, picked].mean(axis=3).reshape(len(errors), -1)  # (S, t 3 × 채널): SCORE_COLUMNS 순서와 같다.


# ===============================
# 3. 분할 전체 측정 (segments.csv에 파일마다 바로 기록)
# ===============================

def select_eval_items():  # eval 파일 목록. EVAL_LIMIT가 있으면 무작위로 그 개수만 고른다.

    items = read_protocol(EVAL_DATA)  # bona fide + spoof 전체

    if EVAL_LIMIT is not None:  # 개발용으로 줄인 경우
        generator = torch.Generator().manual_seed(SEED)  # 언제 돌려도 같은 파일들이 뽑히도록 고정 (지금까지의 8,000개 표본과 같다)
        picked = torch.randperm(len(items), generator=generator)[:EVAL_LIMIT]  # 무작위 순서에서 앞의 EVAL_LIMIT개
        items = [items[i] for i in sorted(picked.tolist())]  # 원래 순서대로 정렬

    return items


def read_done_files(csv_path):  # segments.csv에 이미 기록된 (분할, 파일ID) 목록

    if not os.path.exists(csv_path):  # 처음 실행이면
        return set()

    with open(csv_path, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))  # 첫 줄 = 열 이름
    if header != SEGMENT_COLUMNS:  # 다른 설정(t, 특징)으로 잰 결과와 섞이지 않게 확인한다.
        raise ValueError(f"{csv_path}의 열이 지금 설정과 다르다. 예전 평가 결과이므로 폴더 이름을 바꿔 두고 다시 실행할 것")

    with open(csv_path, "r", encoding="utf-8") as f:
        return {(row["split"], row["file_id"]) for row in csv.DictReader(f)}


def measure_split(model, items, split, stats, fixed_noises, csv_path):  # 파일들을 측정해 segments.csv에 이어서 적는다.

    done = read_done_files(csv_path)  # 이미 측정한 파일
    todo = [item for item in items if (split, item["file_id"]) not in done]  # 아직 측정하지 않은 파일만
    print(f"[{split}] 파일 {len(items)}개 중 이미 측정 {len(items) - len(todo)}개, 이번에 측정 {len(todo)}개")

    is_new_file = not os.path.exists(csv_path)  # 파일이 없으면 헤더부터 쓴다.

    with open(csv_path, "a", newline="", encoding="utf-8") as f:  # 이어 쓰기 모드
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(SEGMENT_COLUMNS)

        for i, item in enumerate(todo):  # 파일마다
            starts, errors = measure_file(model, item["path"], stats, fixed_noises)  # (S, t 수 × 채널 수)

            if i == 0:  # 첫 파일에서 크기와 값을 확인한다. (기본 점수용 (t, 채널) 오차만 보여 준다)
                print(f"  첫 파일 {item['file_id']}: 구간 {errors.shape[0]}개, 오차 열 {errors.shape[1]}개, {dict(zip(SCORE_COLUMNS, np.round(score_errors(errors)[0], 4)))}")

            for start, error in zip(starts, errors):  # 구간마다 한 줄
                start_sec = start * HOP_LENGTH / SAMPLE_RATE  # 시작 프레임 → 초 (무음을 잘라 낸 뒤 기준)
                writer.writerow([split, item["file_id"], item["label"], item["attack_id"], f"{start_sec:.3f}"] + [f"{e:.6f}" for e in error])
            f.flush()  # 파일 하나가 끝날 때마다 디스크에 바로 쓴다. (중간에 멈춰도 남도록)

            if (i + 1) % 1000 == 0:  # 1000개마다 진행 상황
                print(f"  {i + 1}/{len(todo)}")


# ===============================
# 4. 보정과 점수 (dev real만으로 정한다)
# ===============================

def load_segments(csv_path, split, file_ids):  # segments.csv에서 이번에 쓸 구간만 읽는다.

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == split and row["file_id"] in file_ids]

    errors = np.array([[float(row[c]) for c in ERROR_COLUMNS] for row in rows])  # (구간 수, ERROR_COLUMNS 수)

    return rows, score_errors(errors)  # 기본 점수용 (구간 수, SCORE_COLUMNS 수). 나머지 열은 분석 스크립트가 직접 읽는다.


def calibrate(dev_errors):  # dev real 구간의 (t, 채널)별 평균·표준편차

    calibration = {}
    for j, column in enumerate(SCORE_COLUMNS):  # 열마다
        calibration[column] = {
            "mean": float(dev_errors[:, j].mean()),  # μ
            "std": float(dev_errors[:, j].std()),  # σ
        }
    # t마다(t가 작을수록 오차가 크다), 채널마다 오차의 크기가 달라서 그대로 더하면 한 열이 점수를 독차지한다. 그래서 z-점수로 맞춘다.

    return calibration


def segment_scores(errors, calibration):  # 구간 점수 = (t, 채널)별 z-점수의 평균

    z = np.zeros_like(errors)  # (구간 수, 열 수)
    for j, column in enumerate(SCORE_COLUMNS):
        z[:, j] =(errors[:, j] - calibration[column]["mean"]) / calibration[column]["std"]  # z = (e − μ) / σ

    return z.mean(axis=1)  # (구간 수,): 클수록 "real보다 노이즈를 못 맞혔다"


def file_scores(rows, seg_scores):  # 구간 점수 → 파일 점수 (평균, 최댓값)

    files = {}  # 파일ID → 정보와 구간 점수 목록
    for row, score in zip(rows, seg_scores):
        if row["file_id"] not in files:
            files[row["file_id"]] = {"file_id": row["file_id"], "label": row["label"], "attack_id": row["attack_id"], "scores": []}
        files[row["file_id"]]["scores"].append(score)

    results = []
    for info in files.values():  # 파일마다
        results.append({
            "file_id": info["file_id"],
            "label": info["label"],
            "attack_id": info["attack_id"],
            "num_segments": len(info["scores"]),
            "score_mean": float(np.mean(info["scores"])),  # 파일 전체가 얼마나 낯선지
            "score_max": float(np.max(info["scores"])),  # 가장 의심스러운 구간 (부분 변조 대응)
        })

    return results


# ===============================
# 5. 지표
# ===============================

def compute_eer(labels, scores):  # EER: 오탐률(real→fake)과 미탐률(fake→real)이 같아지는 지점의 오류율

    fpr, tpr, _ = roc_curve(labels, scores)  # 임계값을 바꿔 가며 오탐률, 탐지율을 계산한다.
    fnr = 1 - tpr  # 미탐률 = 1 − 탐지율
    idx = np.argmin(np.abs(fnr - fpr))  # 두 값이 가장 가까운 지점

    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(results, key, threshold):  # key: SCORE_KEYS 중 하나

    labels = np.array([1 if r["label"] == "spoof" else 0 for r in results])  # fake=1, real=0
    scores = np.array([r[key] for r in results])  # 클수록 fake로 본다.
    attacks = np.array([r["attack_id"] for r in results])

    metrics = {
        "eer": compute_eer(labels, scores),
        "auroc": float(roc_auc_score(labels, scores)),  # 0.5 = 우연, 0.5보다 작으면 fake의 오차가 오히려 작다는 뜻
        "threshold": float(threshold),
        "false_alarm_rate": float((scores[labels == 0] > threshold).mean()),  # real인데 fake로 판정한 비율 (dev 기준 약 5%가 목표)
        "detection_rate": float((scores[labels == 1] > threshold).mean()),  # fake를 fake로 판정한 비율
        "per_attack": {},
    }

    for attack in sorted(set(attacks[labels == 1])):  # 공격마다: eval real 전체 vs 그 공격의 fake
        use = (labels == 0) | (attacks == attack)
        metrics["per_attack"][str(attack)] = {  # numpy 문자열 → 일반 문자열 (json 저장용)
            "num_spoof": int((attacks == attack).sum()),
            "eer": compute_eer(labels[use], scores[use]),
            "auroc": float(roc_auc_score(labels[use], scores[use])),
        }

    return metrics


# ===============================
# 6. 시험지 (라벨 파일 없는 음성 폴더)
# ===============================

def score_test_samples(model, stats, fixed_noises, calibration, thresholds, out_path):  # 폴더 안 파일마다 점수와 판정

    names = sorted(os.listdir(TEST_SAMPLES_DIR))  # 파일 이름순
    rows = []
    for name in names:
        try:
            _, errors = measure_file(model, os.path.join(TEST_SAMPLES_DIR, name), stats, fixed_noises)
        except ValueError as error:  # 소리가 없는 파일 등 (preprocess.load_audio가 낸 오류)
            print(f"  {name}: 건너뜀 ({error})")
            continue
        seg = segment_scores(score_errors(errors), calibration)  # 구간 점수
        row = {"file": name, "num_segments": len(seg), "score_mean": float(seg.mean()), "score_max": float(seg.max())}
        for key in SCORE_KEYS:
            row[key.replace("score_", "pred_")] = "spoof" if row[key] > thresholds[key] else "bonafide"
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "num_segments", "score_mean", "pred_mean", "score_max", "pred_max"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"시험지 {len(rows)}개 (임계값 score_mean {thresholds['score_mean']:.3f}, score_max {thresholds['score_max']:.3f})")
    for row in rows:
        print(f"  {row['file'][:45]:45s} | mean {row['score_mean']:7.3f} {row['pred_mean']:8s} | max {row['score_max']:7.3f} {row['pred_max']}")


# ===============================
# 7. 저장
# ===============================

def save_predictions(results, thresholds, out_path):  # eval 파일마다 점수와 판정

    pred_columns = [key.replace("score_", "pred_") for key in SCORE_KEYS]  # pred_mean, pred_max
    columns = ["file_id", "label", "attack_id", "num_segments"] + SCORE_KEYS + pred_columns
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in results:
            scores = [f"{r[key]:.6f}" for key in SCORE_KEYS]
            preds = ["spoof" if r[key] > thresholds[key] else "bonafide" for key in SCORE_KEYS]  # 임계값보다 크면 fake
            writer.writerow([r["file_id"], r["label"], r["attack_id"], r["num_segments"]] + scores + preds)


def save_score_hist(results, thresholds, out_path):  # real·fake 점수 분포와 임계값

    fig, axes = plt.subplots(1, len(SCORE_KEYS), figsize=(6 * len(SCORE_KEYS), 4))
    for ax, key in zip(axes, SCORE_KEYS):
        real = [r[key] for r in results if r["label"] == "bonafide"]
        fake = [r[key] for r in results if r["label"] == "spoof"]
        ax.hist(real, bins=100, density=True, alpha=0.6, label=f"bona fide ({len(real)})")  # 개수가 크게 달라서 밀도로 그린다.
        ax.hist(fake, bins=100, density=True, alpha=0.6, label=f"spoof ({len(fake)})")
        ax.axvline(thresholds[key], color="black", linestyle="--", label=f"threshold (dev real {THRESHOLD_PERCENTILE}%)")
        ax.set_title(f"{RUN_NAME}: {key}")
        ax.set_xlabel("score (higher = less real-like)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_table(metrics):  # 결과 표를 출력한다.

    for key in SCORE_KEYS:
        m = metrics[key]
        print(f"{key:10s} | EER {m['eer'] * 100:6.2f}% | AUROC {m['auroc']:.4f} | 오탐률 {m['false_alarm_rate'] * 100:.1f}%, 탐지율 {m['detection_rate'] * 100:.1f}%")

    print("공격별 AUROC (score_mean / score_max)")
    for attack in metrics["score_mean"]["per_attack"]:
        per = [metrics[key]["per_attack"][attack] for key in SCORE_KEYS]
        warn = "  ← < 0.5 (fake의 오차가 더 작음)" if per[0]["auroc"] < 0.5 else ""
        print(f"  {attack} ({per[0]['num_spoof']:5d}개) | {per[0]['auroc']:.4f} / {per[1]['auroc']:.4f}{warn}")


# ===============================
# 8. 실행
# ===============================

if __name__ == "__main__":  # python src/evaluate.py로 실행했을 때만 아래가 동작한다.

    os.makedirs(EVAL_DIR, exist_ok=True)  # 결과 폴더를 만든다.
    segments_path = os.path.join(EVAL_DIR, "segments.csv")
    print(f"장치: {DEVICE}, 실험: {RUN_NAME}, 특징: {FEATURES}, 평가 데이터: {EVAL_DATA}, EVAL_LIMIT: {EVAL_LIMIT}")
    print(f"점수용 t: {SCORE_TIMESTEPS}, 노이즈 seed: {SCORE_NOISE_SEEDS}, 창 2초 / hop {INFER_HOP_SEC}초")

    model = load_model()
    stats = load_stats()  # train real로 계산한 정규화 통계
    fixed_noises = make_fixed_noises()

    # ----- 1. 측정 -----
    dev_items = [item for item in read_protocol(DEV_DATA) if item["label"] == "bonafide"]  # dev는 real만 (real-only)
    eval_items = select_eval_items()
    measure_split(model, dev_items, "dev", stats, fixed_noises, segments_path)
    measure_split(model, eval_items, "eval", stats, fixed_noises, segments_path)

    # ----- 2. 보정 (dev real만) -----
    dev_rows, dev_errors = load_segments(segments_path, "dev", {item["file_id"] for item in dev_items})
    eval_rows, eval_errors = load_segments(segments_path, "eval", {item["file_id"] for item in eval_items})
    print(f"구간 수: dev real {len(dev_rows)}개, eval {len(eval_rows)}개")

    calibration = calibrate(dev_errors)
    for column, value in calibration.items():
        print(f"  {column}: 평균 {value['mean']:.4f}, 표준편차 {value['std']:.4f}")

    # ----- 3. 파일 점수와 임계값 (dev real만) -----
    dev_results = file_scores(dev_rows, segment_scores(dev_errors, calibration))
    eval_results = file_scores(eval_rows, segment_scores(eval_errors, calibration))

    thresholds = {}
    for key in SCORE_KEYS:
        thresholds[key] = float(np.percentile([r[key] for r in dev_results], THRESHOLD_PERCENTILE))  # dev real 95백분위
        print(f"임계값 {key}: {thresholds[key]:.4f}")

    with open(os.path.join(EVAL_DIR, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "columns": calibration, "thresholds": thresholds}, f, indent=2)

    # ----- 4. eval 지표 -----
    metrics = {
        "run_name": RUN_NAME,
        "features": FEATURES,
        "score_timesteps": SCORE_TIMESTEPS,
        "num_files": {
            "eval_bonafide": sum(r["label"] == "bonafide" for r in eval_results),
            "eval_spoof": sum(r["label"] == "spoof" for r in eval_results),
            "dev_bonafide": len(dev_results),
        },
        "eval_limit": EVAL_LIMIT,
    }
    for key in SCORE_KEYS:
        metrics[key] = compute_metrics(eval_results, key, thresholds[key])

    with open(os.path.join(EVAL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_predictions(eval_results, thresholds, os.path.join(EVAL_DIR, "predictions.csv"))
    save_score_hist(eval_results, thresholds, os.path.join(EVAL_DIR, "score_hist.png"))

    print(f"eval 파일: real {metrics['num_files']['eval_bonafide']}개, fake {metrics['num_files']['eval_spoof']}개")
    print_table(metrics)

    # ----- 5. 시험지 -----
    score_test_samples(model, stats, fixed_noises, calibration, thresholds, os.path.join(EVAL_DIR, "test_samples.csv"))
    print(f"평가 끝. 결과 폴더: {EVAL_DIR}")
