# ===============================
# rdm_evaluate.py
# 학습한 RDM으로 파일마다 점수를 내고, dev real로 임계값을 정해 eval을 평가한다. 끝에 시험지(test_samples/)도 판정한다.
#
# 점수 내는 법 (파일 하나)
#   파일 벡터(표준화) x0 → t = 50, 100, 200마다 고정 노이즈 16개를 섞어 x_t → RDM이 노이즈 예측
#   → t마다 (예측 − 정답)²의 평균 = 오차 3개 → dev real 기준 z-점수 3개 → 평균 = 파일 점수 (클수록 real답지 않다)
# 판정: 파일 점수 > dev real 파일 점수의 95백분위 → spoof
#
# 읽는 파일: features/dev.npz, eval.npz (rdm_features.py), checkpoints/best.pt (rdm_train.py)
# 만드는 파일 (outputs/rdm_xlsr_l22/evaluation_<EVAL_DATA> 폴더)
#   calibration.json : dev real로 정한 t별 평균·표준편차와 임계값 (웹 서버도 이 값을 쓴다)
#   metrics.json     : EER, AUROC, 오탐률, 탐지율, 공격별 지표
#   predictions.csv  : eval 파일마다 점수와 판정
#   score_hist.png   : real·fake 점수 분포와 임계값
#   test_samples.csv : 시험지 파일마다 점수와 판정
# ===============================

import csv  # 결과 표를 쓴다.
import json  # 보정값과 지표를 저장한다.
import os  # 파일 경로를 다룬다.

import matplotlib  # 그림을 저장한다.
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt  # 점수 분포를 그린다.
import numpy as np  # 점수를 계산한다.
import torch  # 모델 실행

from config import *  # config.py의 모든 설정값을 가져온다.
from evaluate import compute_metrics  # EER·AUROC·공격별 지표는 스펙트로그램 실험과 똑같이 계산한다.
from preprocess import load_audio  # 시험지 음성 읽기
from rdm_features import file_vector, load_split, load_xlsr  # 파일 벡터 (학습 때와 똑같이)
from rdm_train import CHECKPOINT_DIR_RDM, RDM, standardize  # 모델과 표준화
from train import forward_diffusion  # 노이즈 섞기 (학습 때와 똑같이)

EVAL_DIR_RDM = os.path.join(RDM_DIR, f"evaluation_{EVAL_DATA}")  # 평가 결과 폴더


# ===============================
# 1. 준비: 모델, 고정 노이즈
# ===============================

def load_rdm():  # best.pt → (모델, 표준화 평균, 표준편차)

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR_RDM, "best.pt"), weights_only=False)  # numpy 통계가 함께 있어서 weights_only=False
    if checkpoint["layer"] != XLSR_LAYER:  # 다른 층으로 학습한 모델이면 벡터 뜻이 달라서 멈춘다.
        raise ValueError(f"모델은 {checkpoint['layer']}층으로 학습했는데 config는 {XLSR_LAYER}층이다.")
    model = RDM().to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # Dropout을 끈다. (늘 같은 점수가 나오도록)
    print(f"RDM: epoch {checkpoint['epoch']}, dev real 손실 {checkpoint['val_loss']:.4f}")

    return model, checkpoint["mean"], checkpoint["std"]


def make_fixed_noises():  # 모든 파일이 똑같은 노이즈 16개로 시험을 본다. (파일마다 운이 달라지지 않도록)

    generator = torch.Generator().manual_seed(SEED)
    return torch.randn((RDM_NOISE_COUNT, XLSR_DIM), generator=generator).to(DEVICE)  # (16, 1024)


# ===============================
# 2. 오차 측정과 점수
# ===============================

def measure_errors(model, x0, noises):  # 표준화한 벡터 (N, 1024) → t별 오차 (N, t 수). 노이즈 16개의 평균

    x0 = torch.tensor(x0, dtype=torch.float32, device=DEVICE)
    errors = np.zeros((len(x0), len(SCORE_TIMESTEPS)))  # (N, 3)
    with torch.no_grad():
        for j, t_value in enumerate(SCORE_TIMESTEPS):  # t = 50, 100, 200
            t = torch.full((len(x0),), t_value, device=DEVICE)  # 모든 파일에 같은 t
            for noise in noises:  # 노이즈 16개
                noise = noise.expand_as(x0)  # (1024,) → (N, 1024): 모든 파일에 같은 노이즈
                x_t, _ = forward_diffusion(x0, t, noise)  # 노이즈를 섞는다.
                error = ((model(x_t, t) - noise) ** 2).mean(dim=1)  # (N,): 파일마다 (예측 − 정답)²의 평균
                errors[:, j] += error.cpu().numpy() / len(noises)  # 16개 평균

    return errors


def calibrate(dev_errors):  # dev real 오차의 t별 평균·표준편차 (t마다 오차 크기가 달라서 맞춘다)

    return {"mean": dev_errors.mean(axis=0).tolist(), "std": dev_errors.std(axis=0).tolist()}


def file_scores(errors, calibration):  # (N, t 수) → (N,): t별 z-점수의 평균. 0 = 보통 dev real, 클수록 real답지 않다.

    z = (errors - np.array(calibration["mean"])) / np.array(calibration["std"])

    return z.mean(axis=1)


# ===============================
# 3. 시험지 (라벨 파일 없는 음성 폴더)
# ===============================

def score_test_samples(model, mean, std, noises, calibration, out_path):  # 폴더 안 파일마다 점수와 판정

    extractor, xlsr = load_xlsr()  # 시험지는 벡터가 저장되어 있지 않아서 여기서 XLS-R로 뽑는다.
    rows = []
    for name in sorted(os.listdir(TEST_SAMPLES_DIR)):  # 파일 이름순
        try:
            wave = load_audio(os.path.join(TEST_SAMPLES_DIR, name))  # 학습과 같은 음성 읽기
        except ValueError as error:  # 소리가 없는 파일 (load_audio가 낸 오류)
            print(f"  {name}: 건너뜀 ({error})")
            continue
        x0 = standardize(file_vector(wave, extractor, xlsr)[None], mean, std)  # (1, 1024)
        score = float(file_scores(measure_errors(model, x0, noises), calibration)[0])
        rows.append({"file": name, "seconds": round(len(wave) / SAMPLE_RATE, 1), "score": round(score, 3),
                     "pred": "spoof" if score > calibration["threshold"] else "bonafide"})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "seconds", "score", "pred"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n시험지 {len(rows)}개 (임계값 {calibration['threshold']:.3f})")
    for row in rows:
        print(f"  {row['file'][:50]:50s} {row['seconds']:5.1f}초 | 점수 {row['score']:7.3f} → {row['pred']}")


# ===============================
# 4. 저장
# ===============================

def save_predictions(data, scores, threshold, out_path):  # eval 파일마다 점수와 판정

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "label", "attack_id", "score", "pred"])
        for file_id, label, attack, score in zip(data["file_id"], data["label"], data["attack_id"], scores):
            writer.writerow([file_id, label, attack, f"{score:.6f}", "spoof" if score > threshold else "bonafide"])


def save_score_hist(scores, labels, threshold, out_path):  # real·fake 점수 분포와 임계값

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[labels == "bonafide"], bins=100, density=True, alpha=0.6, label=f"bona fide ({(labels == 'bonafide').sum()})")
    ax.hist(scores[labels == "spoof"], bins=100, density=True, alpha=0.6, label=f"spoof ({(labels == 'spoof').sum()})")  # 개수가 달라서 밀도로
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold (dev real {THRESHOLD_PERCENTILE}%)")
    ax.set_title(f"{RDM_RUN_NAME}: {EVAL_DATA}")
    ax.set_xlabel("score (higher = less real-like)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ===============================
# 5. 실행
# ===============================

if __name__ == "__main__":

    os.makedirs(EVAL_DIR_RDM, exist_ok=True)
    model, mean, std = load_rdm()
    noises = make_fixed_noises()

    # ----- dev real: 보정과 임계값 (fake는 쓰지 않는다) -----
    dev_errors = measure_errors(model, standardize(load_split("dev")["vectors"], mean, std), noises)  # (2548, 3)
    calibration = calibrate(dev_errors)
    dev_scores = file_scores(dev_errors, calibration)
    calibration["threshold"] = float(np.percentile(dev_scores, THRESHOLD_PERCENTILE))  # dev real 95백분위
    with open(os.path.join(EVAL_DIR_RDM, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    print(f"dev real 오차 t{SCORE_TIMESTEPS}: 평균 {np.round(calibration['mean'], 4)}, 임계값 {calibration['threshold']:.3f}")

    # ----- eval: 여기서 처음 fake를 본다 -----
    data = load_split("eval")
    eval_errors = measure_errors(model, standardize(data["vectors"], mean, std), noises)
    scores = file_scores(eval_errors, calibration)
    results = [{"label": l, "attack_id": a, "score": s} for l, a, s in zip(data["label"], data["attack_id"], scores)]
    metrics = compute_metrics(results, "score", calibration["threshold"])
    with open(os.path.join(EVAL_DIR_RDM, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    save_predictions(data, scores, calibration["threshold"], os.path.join(EVAL_DIR_RDM, "predictions.csv"))
    save_score_hist(scores, data["label"], calibration["threshold"], os.path.join(EVAL_DIR_RDM, "score_hist.png"))

    print(f"\n{EVAL_DATA} ({len(scores)}개) | EER {metrics['eer'] * 100:.2f}% | AUROC {metrics['auroc']:.4f} | "
          f"오탐률 {metrics['false_alarm_rate'] * 100:.1f}% | 탐지율 {metrics['detection_rate'] * 100:.1f}%")
    for attack, m in metrics["per_attack"].items():
        warn = "  ← < 0.5 (fake의 오차가 더 작음)" if m["auroc"] < 0.5 else ""
        print(f"  {attack} ({m['num_spoof']:5d}개) | AUROC {m['auroc']:.4f}{warn}")

    # ----- 시험지 -----
    score_test_samples(model, mean, std, noises, calibration, os.path.join(EVAL_DIR_RDM, "test_samples.csv"))
