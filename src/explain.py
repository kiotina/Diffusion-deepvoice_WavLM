# ===============================
# explain.py
# 웹 데모용: 음성 하나를 RDM(XLS-R 파일 벡터 위의 디퓨전)으로 판정하고, 판정 근거를 만든다.
#
# 판정: evaluate.py와 똑같은 계산이다.
#   음성 → 앞뒤 무음 자름 → 최대 진폭 1.0 → XLS-R 22층 프레임 → 파일 평균 벡터 → 표준화
#   → t = 50, 100, 200마다 고정 노이즈 16개를 섞고 RDM이 맞힌 오차 → dev real 기준 z → 평균 = 평균 이상도 → 임계값과 비교
#
# 근거 (모두 판정에 쓴 값과 같은 계산에서 나온다. 별도의 설명 모델은 없다)
#   ① 평균 이상도 분포: dev real 점수 분포(임계값 기준) + eval fake 점수 분포(화면 비교용) 위에 이 파일의 위치
#   ② 분포 거리(참고): t = 50, 200 두 오차(z)를 점 하나로 보고, dev real 무리에서 얼마나 떨어졌는지(마할라노비스)
#                      산점도에 dev real 점, eval fake 점, real 90% 타원, 이 파일을 함께 그린다. 판정에는 쓰지 않는다.
#   ③ 구간별 이상도: 2초 창(1초 간격)마다 그 창의 프레임만 평균 낸 벡터로 같은 점수를 낸다. "어느 구간이 낯선가"
#                    창은 파일보다 짧아서 점수가 더 흔들리므로, 기준(평균·표준편차·임계값)을 dev real 창으로 따로 정한다.
#
# fake는 학습·임계값 어디에도 쓰지 않는다. 기준값을 만들 때 eval fake 점수를 계산하지만 화면에 비교용으로 그리는 데만 쓴다.
#
# 사용법
#   python src/explain.py              기준값을 만든다. (best.pt가 바뀔 때마다 한 번, 약 2~3분)
#   python src/explain.py 음성파일경로    파일 하나를 분석해 요약을 출력한다. (웹 없이 확인용)
#
# 읽는 파일: outputs/<RUN_NAME>/checkpoints/best.pt, evaluation_<EVAL_DATA>/calibration.json (evaluate.py),
#           outputs/rdm_features_xlsr_l22/ 의 DEV_DATA·EVAL_DATA 벡터 (preprocess.py)
# 만드는 파일: outputs/<RUN_NAME>/explain/reference.json (창 기준값, 점수 분포, 분포 거리 기준, 산점도 점)
# ===============================

import json  # 기준값 읽고 쓰기
import os  # 파일 경로
import sys  # 명령줄 인자
import time  # 기준값을 만든 시각

import librosa  # 화면용 Mel 그림 (모델 입력과 무관)
import numpy as np  # 계산
import torch  # 모델 실행

from config import *  # config.py의 모든 설정값
from evaluate import load_rdm, make_fixed_noises, measure_errors  # 점수 계산 (평가와 똑같이)
from preprocess import frame_features, load_audio, load_many, load_xlsr, normalize_peak, pool, select_bonafide, trim_silence  # 음성 읽기·XLS-R 특징 (학습과 똑같이)
from train import standardize  # 표준화

EXPLAIN_DIR = os.path.join(RUN_DIR, "explain")
REFERENCE_PATH = os.path.join(EXPLAIN_DIR, "reference.json")
REFERENCE_VERSION = 2  # 기준값 파일 형식. 형식이 바뀌면 웹 서버가 다시 만들라고 알린다.
WINDOW_FRAMES = int(round(EXPLAIN_WINDOW_SEC / XLSR_FRAME_SEC))  # 2초 = 100프레임
HOP_FRAMES = int(round(EXPLAIN_HOP_SEC / XLSR_FRAME_SEC))  # 1초 = 50프레임
DIST_IDX = [SCORE_TIMESTEPS.index(t) for t in DIST_TIMESTEPS]  # 분포 거리에 쓸 t의 위치 (예: [0, 2])
HIST_BINS = 50  # 평균 이상도 분포 그림의 막대 수
LOSSY_CODECS = {"mp3", "aac", "opus", "vorbis", "amr_nb", "amr_wb", "wmav2"}  # 손실 압축 코덱 (입력 점검 안내용)


# ===============================
# 1. 공통 계산
# ===============================

def to_z(errors, calibration):  # 파일 오차 (N, t 수) → t별 z (dev real 파일 기준)

    return (errors - np.array(calibration["mean"])) / np.array(calibration["std"])


def mahalanobis(points, center, inv_cov):  # (N, 2) → (N,): 진짜 음성 무리 모양(공분산)을 고려한 거리

    d = points - center
    return np.sqrt(np.einsum("ni,ij,nj->n", d, inv_cov, d))


def ellipse_points(center, cov, radius, num=120):  # 마할라노비스 거리가 radius인 점들 = 타원 (그림용)

    values, vectors = np.linalg.eigh(cov)  # 타원의 축 방향과 길이
    theta = np.linspace(0, 2 * np.pi, num + 1)
    circle = np.stack([np.cos(theta), np.sin(theta)])  # (2, num+1)
    points = center[:, None] + radius * (vectors @ (np.sqrt(values)[:, None] * circle))

    return points.T  # (num+1, 2)


# ===============================
# 2. 설명용 창
# ===============================

def window_starts(num_frames):  # 2초 창의 시작 프레임 목록. 2초보다 짧으면 창 하나(파일 전체)

    if num_frames <= WINDOW_FRAMES:
        return [0]
    starts = list(range(0, num_frames - WINDOW_FRAMES + 1, HOP_FRAMES))
    if starts[-1] + WINDOW_FRAMES < num_frames:  # 끝부분이 남으면 끝에 맞춘 창을 하나 더 (마지막 소리도 보도록)
        starts.append(num_frames - WINDOW_FRAMES)

    return starts


def window_vectors(frames):  # 프레임 특징 (T, 1024) → 창마다 평균 벡터 (창 수, 1024), 시작 프레임 목록

    starts = window_starts(frames.shape[0])
    vectors = np.stack([pool(frames[s:s + WINDOW_FRAMES]) for s in starts])

    return vectors, starts


def window_z(errors, reference):  # 창 오차 (창 수, t 수) → t별 z (dev real 창 기준)

    return (errors - np.array(reference["window_mean"])) / np.array(reference["window_std"])


# ===============================
# 3. 기준값 만들기 (python src/explain.py)
# ===============================

def build_reference():

    model, mean, std = load_rdm()
    noises = make_fixed_noises()
    extractor, xlsr = load_xlsr()
    calibration = json.load(open(os.path.join(EVAL_DIR, "calibration.json"), encoding="utf-8"))  # 파일 점수 기준 (evaluate.py)
    rng = np.random.default_rng(SEED)

    # ----- dev real 파일: 평균 이상도 분포, 분포 거리 기준 (임계값과 같은 데이터) -----
    dev = load_many(DEV_DATA)
    dev_z = to_z(measure_errors(model, standardize(dev["vectors"], mean, std), noises), calibration)  # (dev 수, 3)
    dev_scores = dev_z.mean(axis=1)

    # ----- eval fake 파일: 화면 비교용으로만 (학습·임계값에는 쓰지 않는다) -----
    ev = load_many([EVAL_DATA])
    fake = ev["label"] == "spoof"
    fake_z = to_z(measure_errors(model, standardize(ev["vectors"][fake], mean, std), noises), calibration)
    fake_scores = fake_z.mean(axis=1)

    # ----- 평균 이상도 분포 그림: real과 fake를 같은 막대 구간으로 -----
    lo, hi = np.percentile(np.concatenate([dev_scores, fake_scores]), [0.5, 99.5])  # 아주 드문 끝값은 양 끝 막대에 모은다.
    edges = np.linspace(lo, hi, HIST_BINS + 1)
    hist = {"edges": edges.tolist(),
            "real": np.histogram(np.clip(dev_scores, lo, hi), bins=edges)[0].tolist(),
            "fake": np.histogram(np.clip(fake_scores, lo, hi), bins=edges)[0].tolist()}

    # ----- 분포 거리: t 두 개의 z를 점 하나로 보고, dev real 무리의 모양(공분산)으로 거리를 잰다 -----
    dev_points = dev_z[:, DIST_IDX]  # (dev 수, 2)
    center = dev_points.mean(axis=0)
    cov = np.cov(dev_points, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    dev_dist = mahalanobis(dev_points, center, inv_cov)
    dist_threshold = float(np.percentile(dev_dist, THRESHOLD_PERCENTILE))  # dev real의 90%가 이 거리 안에 든다.

    def sample(points):  # 산점도용 점 (너무 많으면 무작위로 일부만)
        picked = rng.choice(len(points), size=min(DISPLAY_POINTS, len(points)), replace=False)
        return np.round(points[picked], 2).tolist()

    # ----- dev real 일부의 창 오차 → 창 기준값 -----
    items = [it for name in DEV_DATA for it in select_bonafide(name)]
    picked = rng.choice(len(items), size=min(EXPLAIN_REF_FILES, len(items)), replace=False)  # 언어·데이터가 섞이도록 무작위
    window_errors, file_index = [], []
    for k, i in enumerate(picked):
        frames = frame_features(load_audio(items[i]["path"]), extractor, xlsr)
        vectors, _ = window_vectors(frames)
        window_errors.append(measure_errors(model, standardize(vectors, mean, std), noises))
        file_index += [k] * len(vectors)
        if (k + 1) % 200 == 0:
            print(f"  창 기준값: {k + 1}/{len(picked)}")
    window_errors = np.concatenate(window_errors)  # (창 수, t 수)
    file_index = np.array(file_index)
    reference = {"window_mean": window_errors.mean(0).tolist(), "window_std": window_errors.std(0).tolist()}
    window_scores = window_z(window_errors, reference).mean(axis=1)
    max_per_file = np.array([window_scores[file_index == k].max() for k in range(len(picked))])  # 파일마다 "가장 낯선 창"
    reference["window_threshold"] = float(np.percentile(max_per_file, THRESHOLD_PERCENTILE))  # dev real 파일의 90%는 이 선을 넘는 창이 없다.

    reference.update({
        "dev_scores": dev_scores.tolist(),
        "hist": hist,
        "distance": {"timesteps": DIST_TIMESTEPS, "center": center.tolist(), "cov": cov.tolist(), "inv_cov": inv_cov.tolist(),
                     "threshold": dist_threshold, "dev_dist": np.sort(dev_dist).tolist(),
                     "ellipse": np.round(ellipse_points(center, cov, dist_threshold), 3).tolist()},
        "scatter": {"real": sample(dev_points), "fake": sample(fake_z[:, DIST_IDX])},
        "meta": {"version": REFERENCE_VERSION, "run_name": RUN_NAME,
                 "checkpoint_step": int(torch.load(os.path.join(RUN_DIR, "checkpoints", "best.pt"), weights_only=False)["step"]),
                 "threshold": calibration["threshold"], "num_dev_files": len(dev_scores), "num_fake_files": int(fake.sum()),
                 "eval_data": EVAL_DATA, "num_window_files": len(picked), "num_windows": len(window_errors),
                 "created": time.strftime("%Y-%m-%d %H:%M")},
    })
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    with open(REFERENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(reference, f)
    print(f"창 {len(window_errors)}개 (dev real {len(picked)}개 파일): 창 임계값 {reference['window_threshold']:.3f}")
    print(f"평균 이상도: dev real {len(dev_scores)}개, 임계값 {calibration['threshold']:.3f} | 화면 비교용 eval fake {int(fake.sum())}개")
    print(f"분포 거리 (t={DIST_TIMESTEPS}): 임계값 {dist_threshold:.3f}, "
          f"eval fake 중 임계값 넘음 {np.mean(mahalanobis(fake_z[:, DIST_IDX], center, inv_cov) > dist_threshold) * 100:.1f}% (참고)")
    print(f"저장 {REFERENCE_PATH}")


# ===============================
# 4. 입력 점검 (real-only 모델은 낯선 조건을 모두 "낯설다"로 볼 수 있어서 따로 알린다)
# ===============================

def inspect_audio(wave, source_info, original_sec, truncated):  # wave: 무음을 자르고 크기를 맞추기 전

    source_info = source_info or {}
    notes = []  # {"level": "warn" 또는 "info", "text": 문장}
    peak = float(np.abs(wave).max())
    rms = librosa.feature.rms(y=wave, frame_length=WIN_LENGTH, hop_length=WIN_LENGTH // 2)[0]  # 25ms마다 소리 크기
    silence_ratio = float((rms < peak * 10 ** (SILENCE_DB_THRESHOLD / 20)).mean())  # 가장 큰 소리보다 40dB 작은 구간 비율 (가운데 쉼)
    clip_ratio = float((np.abs(wave) >= 0.999).mean())
    codec, ext, sample_rate = (source_info.get("codec") or "").lower(), (source_info.get("ext") or "").lower(), source_info.get("sample_rate")

    if truncated:
        notes.append({"level": "info", "text": f"{original_sec:.1f}초 중 앞 {MAX_AUDIO_SEC:g}초만 분석했습니다."})
    if sample_rate and sample_rate < SAMPLE_RATE:
        notes.append({"level": "warn", "text": f"원래 샘플레이트가 {sample_rate} Hz입니다. {sample_rate / 2000:g} kHz 이상 정보가 비어 있어 진짜 음성도 점수가 높게 나올 수 있습니다."})
    if codec in LOSSY_CODECS or (not codec and ext in {"mp3", "m4a", "aac", "ogg", "opus", "webm", "amr", "wma"}):
        notes.append({"level": "info", "text": f"손실 압축 형식({codec or ext})입니다. 학습 데이터는 대부분 무손실(FLAC·WAV)이었습니다."})
    if clip_ratio > 0.001:
        notes.append({"level": "warn", "text": f"소리가 깨진(클리핑) 샘플이 {clip_ratio * 100:.2f}% 있습니다."})
    if peak < 0.05:
        notes.append({"level": "warn", "text": f"소리가 매우 작습니다(최대 진폭 {peak:.3f}). 크기는 맞춰서 분석하지만 배경 잡음도 함께 커집니다."})
    if len(wave) < EXPLAIN_WINDOW_SEC * SAMPLE_RATE:  # 길이별 분석(2026-11-25): 1.5초 미만 dev real 오탐 16.9%, 2초 이상은 3~4%
        notes.append({"level": "warn", "text": f"말소리가 {len(wave) / SAMPLE_RATE:.1f}초로 짧습니다. 2초보다 짧으면 진짜 음성도 점수가 높게 나오기 쉽습니다(1.5초 미만 진짜 음성 오탐 약 17%)."})
    if silence_ratio > 0.5:
        notes.append({"level": "warn", "text": f"쉬는 구간이 {silence_ratio * 100:.0f}%로 많아 말소리가 적습니다."})

    quality = {"peak": peak, "clip_ratio": clip_ratio, "silence_ratio": silence_ratio, "sample_rate": sample_rate,
               "channels": source_info.get("channels"), "format": codec or ext or None}

    return quality, notes


# ===============================
# 5. 판정과 설명
# ===============================

def make_summary(r):  # 숫자로부터 규칙에 따라 문장을 만든다. (생성형 모델은 쓰지 않는다)

    lines = []
    if r["verdict"] == "spoof":
        lines.append(f"평균 이상도가 {r['score']:.2f}로 임계값 {r['threshold']:.2f}보다 높아 딥보이스로 의심됩니다.")
    else:
        lines.append(f"평균 이상도가 {r['score']:.2f}로 임계값 {r['threshold']:.2f} 이하라서 진짜 음성 범위로 판단했습니다.")
    lines.append(f"개발용 진짜 음성 {r['reference']['num_dev_files']:,}개 중 {r['percentile']:.1f}%가 이 파일보다 평균 이상도가 낮습니다.")

    d = r["distance"]
    place = "밖" if d["flag"] else "안"
    lines.append(f"분포 거리(참고)는 {d['value']:.2f}로, 진짜 음성 90%가 들어가는 범위(기준 {d['threshold']:.2f}) {place}에 있습니다.")

    top = max(r["segments"], key=lambda s: s["score"])
    if r["flagged_intervals"]:
        spans = ", ".join(f"{a:.1f}–{b:.1f}초" for a, b in r["flagged_intervals"])
        lines.append(f"2초 구간 기준을 넘은 곳은 {spans}이고, 가장 낯선 구간은 {top['start']:.1f}–{top['end']:.1f}초입니다.")
    else:
        lines.append(f"2초 구간 기준을 넘은 곳은 없습니다. 가장 낯선 구간은 {top['start']:.1f}–{top['end']:.1f}초입니다.")

    num_warn = sum(note["level"] == "warn" for note in r["notes"])
    if num_warn:
        lines.append(f"입력 점검 경고가 {num_warn}개 있습니다. 녹음 조건이 학습 데이터와 다르면 진짜 음성도 점수가 높아질 수 있습니다.")

    return lines


class Explainer:  # 서버를 켤 때 한 번 만들고, 요청마다 analyze_wave()를 부른다.

    def __init__(self):

        if not os.path.exists(REFERENCE_PATH):
            raise FileNotFoundError(f"기준값 파일이 없습니다: {REFERENCE_PATH}\n먼저 python src/explain.py 를 실행하세요.")
        self.reference = json.load(open(REFERENCE_PATH, encoding="utf-8"))
        self.calibration = json.load(open(os.path.join(EVAL_DIR, "calibration.json"), encoding="utf-8"))
        self.model, self.mean, self.std = load_rdm()
        self.checkpoint = torch.load(os.path.join(RUN_DIR, "checkpoints", "best.pt"), weights_only=False)
        self.noises = make_fixed_noises()
        self.extractor, self.xlsr = load_xlsr()

        meta = self.reference.get("meta", {})  # 기준값을 만든 뒤 모델·임계값·형식이 바뀌었으면 점수가 조용히 틀어진다. 그래서 멈춘다.
        if (meta.get("version") != REFERENCE_VERSION or meta.get("run_name") != RUN_NAME or meta.get("checkpoint_step") != self.checkpoint["step"]
                or abs(meta.get("threshold", float("nan")) - self.calibration["threshold"]) > 1e-9
                or self.reference["distance"]["timesteps"] != DIST_TIMESTEPS):
            raise RuntimeError("기준값 파일이 지금 모델·설정과 다릅니다. python src/explain.py 를 다시 실행하세요.")

        self.dev_scores = np.array(self.reference["dev_scores"])
        dist = self.reference["distance"]
        self.dist_center, self.dist_inv_cov = np.array(dist["center"]), np.array(dist["inv_cov"])
        self.dist_dev = np.array(dist["dev_dist"])
        self.mel_hz = librosa.mel_frequencies(n_mels=DISPLAY_N_MELS + 2, fmin=0.0, fmax=SAMPLE_RATE / 2)[1:-1]  # 화면용 Mel 칸의 중심 주파수

    def model_card(self):  # 화면과 보고서에 적을 모델 정보

        return {
            "run_name": RUN_NAME,
            "feature_type": "xlsr",
            "feature": f"XLS-R 300M {XLSR_LAYER}층 특징의 파일 평균 벡터({XLSR_DIM}차원)",
            "network": f"MLP 디퓨전(DDPM, 잔차 블록 {BLOCKS}개)",
            "checkpoint_epoch": f"step {self.checkpoint['step']}",  # 화면이 쓰는 이름을 그대로 둔다. (걸음 수)
            "train_data": " + ".join(TRAIN_DATA) + (" (방 환경·배속 증강)" if AUGMENT else ""),
            "dev_data": " + ".join(DEV_DATA),
            "eval_data": self.reference["meta"]["eval_data"],
            "score_timesteps": SCORE_TIMESTEPS,
            "dist_timesteps": DIST_TIMESTEPS,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "num_dev_files": self.reference["meta"]["num_dev_files"],
            "reference_created": self.reference["meta"]["created"],
        }

    def info(self):  # GET /api/info
        return {"model": self.model_card(), "threshold": self.calibration["threshold"], "max_audio_sec": MAX_AUDIO_SEC, "max_upload_mb": MAX_UPLOAD_MB}

    def analyze(self, path, source_info=None):  # 파일 경로 → 결과 (명령줄 확인용). load_audio가 이미 자르고 맞췄으므로 그대로 넘긴다.
        return self.analyze_wave(load_audio(path), source_info)

    def analyze_wave(self, wave, source_info=None):  # 16kHz 모노 waveform → 판정과 설명 (dict)

        original_sec = len(wave) / SAMPLE_RATE
        peak = float(np.abs(wave).max()) if len(wave) else 0.0
        if peak < MIN_PEAK:  # load_audio와 같은 규칙: 거의 무음인 파일은 분석하지 않는다.
            raise ValueError("소리가 없는 파일입니다. 분석할 수 없습니다.")

        wave, start = trim_silence(wave)  # 학습 때처럼 앞뒤 무음을 잘라 낸다.
        trim_offset = start / SAMPLE_RATE  # 웹에서 재생 위치를 맞출 때 쓴다.
        truncated = len(wave) > MAX_AUDIO_SEC * SAMPLE_RATE
        wave = wave[:int(MAX_AUDIO_SEC * SAMPLE_RATE)]  # 너무 길면 앞부분만 (GPU 메모리, 응답 시간)
        if len(wave) < MIN_AUDIO_SEC * SAMPLE_RATE:
            raise ValueError(f"말소리가 너무 짧습니다. 앞뒤 무음을 빼고 {MIN_AUDIO_SEC:g}초 이상이어야 합니다.")

        quality, notes = inspect_audio(wave, source_info, original_sec, truncated)
        quality["duration"] = len(wave) / SAMPLE_RATE
        notes.append({"level": "info", "text": f"학습 때처럼 앞뒤 무음을 잘라 내고 소리 크기를 맞춰 {quality['duration']:.1f}초를 분석했습니다. (원래 {original_sec:.1f}초)"})
        wave = normalize_peak(wave).astype(np.float32)

        # ----- 판정: 파일 전체 벡터 (evaluate.py와 같은 계산) -----
        frames = frame_features(wave, self.extractor, self.xlsr)  # (T, 1024)
        file_x = standardize(pool(frames)[None], self.mean, self.std)  # (1, 1024)
        file_z = to_z(measure_errors(self.model, file_x, self.noises), self.calibration)[0]  # (3,)
        score = float(file_z.mean())
        threshold = self.calibration["threshold"]

        # ----- 근거: 분포 거리 (참고, 판정에 쓰지 않음) -----
        point = file_z[DIST_IDX]
        dist_value = float(mahalanobis(point[None], self.dist_center, self.dist_inv_cov)[0])
        dist_threshold = self.reference["distance"]["threshold"]

        # ----- 근거: 2초 창마다 같은 점수 -----
        vectors, starts = window_vectors(frames)
        win_z = window_z(measure_errors(self.model, standardize(vectors, self.mean, self.std), self.noises), self.reference)  # (창 수, 3)
        win_scores = win_z.mean(axis=1)
        T = frames.shape[0]
        segments, sums, counts = [], np.zeros((len(SCORE_TIMESTEPS), T)), np.zeros(T)
        for s, sc, z in zip(starts, win_scores, win_z):
            e = min(s + WINDOW_FRAMES, T)
            segments.append({"start": round(s * XLSR_FRAME_SEC, 3), "end": round(e * XLSR_FRAME_SEC, 3), "score": float(sc),
                             "flagged": bool(sc > self.reference["window_threshold"])})
            sums[:, s:e] += z[:, None]  # 프레임마다 그 프레임을 덮는 창들의 z를 모은다.
            counts[s:e] += 1
        frame_z = sums / np.maximum(counts, 1)  # (3, T): 겹친 창의 평균
        flagged = [[seg["start"], seg["end"]] for seg in segments if seg["flagged"]]
        merged = []
        for a, b in flagged:  # 겹치는 구간을 하나로 합친다.
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])

        # ----- 화면용 Mel (모델 입력 아님) -----
        mel = librosa.feature.melspectrogram(y=wave, sr=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=320, n_mels=DISPLAY_N_MELS)
        mel_db = np.round(10 * np.log10(mel + 1e-10)).astype(int)  # 정수 dB (전송량을 줄이고 세부를 덜 담는다)

        meta = self.reference["meta"]
        result = {
            "verdict": "spoof" if score > threshold else "bonafide",
            "score": score, "threshold": threshold, "percentile": float((self.dev_scores < score).mean() * 100),
            "distance": {"value": dist_value, "threshold": dist_threshold, "flag": dist_value > dist_threshold,
                         "percentile": float((self.dist_dev < dist_value).mean() * 100),
                         "point": [float(v) for v in point], "timesteps": DIST_TIMESTEPS},
            "duration": quality["duration"], "original_duration": original_sec, "trim_offset": trim_offset,
            "segment_sec": EXPLAIN_WINDOW_SEC, "segment_threshold": self.reference["window_threshold"],
            "segments": segments, "flagged_intervals": merged,
            "frames": {"frame_sec": XLSR_FRAME_SEC, "offset_sec": 0.0, "num_frames": int(T),
                       "z_t": np.round(frame_z, 2).tolist(), "covered": (counts > 0).astype(int).tolist()},
            "display": {"frame_sec": 320 / SAMPLE_RATE, "mel_hz": [round(float(v)) for v in self.mel_hz], "mel_db": mel_db.tolist()},
            "reference": {"num_dev_files": meta["num_dev_files"], "num_fake_files": meta["num_fake_files"], "eval_data": meta["eval_data"],
                          "hist": self.reference["hist"],
                          "scatter": {**self.reference["scatter"], "ellipse": self.reference["distance"]["ellipse"]}},
            "model": self.model_card(), "quality": quality, "notes": notes,
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
        d = result["distance"]
        print(f"판정: {'딥보이스 의심' if result['verdict'] == 'spoof' else '진짜 음성 범위'} (평균 이상도 {result['score']:.3f}, 임계값 {result['threshold']:.3f})")
        print(f"분포 거리(참고): {d['value']:.3f} (기준 {d['threshold']:.3f}) {'범위 밖' if d['flag'] else '범위 안'}")
        print("2초 구간: " + ", ".join(f"{s['start']:.0f}–{s['end']:.1f}초 {s['score']:+.2f}{'*' if s['flagged'] else ''}" for s in result["segments"]))
        for line in result["summary"]:
            print(f"  - {line}")
        for note in result["notes"]:
            print(f"  [{note['level']}] {note['text']}")
