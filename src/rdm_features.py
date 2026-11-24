# ===============================
# rdm_features.py
# 음성 파일 하나 → XLS-R 22층 특징 → 프레임 평균 = 파일 벡터 하나(1024)로 바꿔서 저장한다.
# 이 벡터가 RDM 디퓨전(rdm_train.py)의 입력이다. XLS-R은 학습하지 않고 숫자로 바꾸는 일만 한다.
#
# 만드는 파일 (outputs/rdm_xlsr_l22/features 폴더)
#   train.npz : train real 파일 벡터 (학습용)
#   dev.npz   : dev real 파일 벡터 (조기 종료·점수 보정·임계값용)
#   eval.npz  : eval 파일 벡터 (EVAL_LIMIT개 표본, real과 fake. 평가에서 처음 fake를 본다)
#   npz 안: vectors (파일 수, 1024), file_id, label, attack_id, speaker_id
#
# file_vector 함수는 rdm_evaluate.py(시험지)와 웹 서버에서도 그대로 쓴다. → 학습과 판정이 똑같은 벡터를 본다.
# ===============================

import os  # 파일 경로를 다룬다.

import numpy as np  # 벡터를 저장한다.
import torch  # XLS-R을 GPU에서 실행한다.
from transformers import AutoFeatureExtractor, AutoModel  # Hugging Face에서 XLS-R을 불러온다.

from config import *  # config.py의 모든 설정값을 가져온다.
from evaluate import select_eval_items  # 지금까지 비교에 쓴 eval 8,000개 표본 (seed 고정)
from preprocess import load_audio, select_bonafide  # 음성 읽기(무음 파일 거르기·무음 자르기·크기 맞추기)를 다른 실험과 똑같이

FEATURE_DIR = os.path.join(RDM_DIR, "features")  # 파일 벡터를 저장할 폴더


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

    x = torch.from_numpy(wave)[None].to(DEVICE)  # (1, 샘플 수): 배치 1개
    with torch.no_grad():  # 학습하지 않으므로 gradient를 계산하지 않는다. (메모리 절약)
        out = model(x, output_hidden_states=True)  # 모든 층의 출력을 받는다.
    frames = out.hidden_states[XLSR_LAYER][0]  # (프레임 수, 1024): 22층 출력. 20ms마다 벡터 하나 (0번은 CNN 출력, 1~24번이 Transformer 층)

    return frames.mean(dim=0).float().cpu().numpy()  # 프레임 평균 → (1024,): "이 파일 전체의 성질"만 남긴다.


# ===============================
# 3. 파일 목록 → 벡터 묶음 저장
# ===============================

def build_split(items, name, extractor, model):  # items: read_protocol 형식의 파일 목록. name: "train", "dev", "eval"

    out_path = os.path.join(FEATURE_DIR, f"{name}.npz")
    if os.path.exists(out_path):  # 이미 만들어 두었으면 다시 하지 않는다. (다시 만들려면 파일을 지운다)
        print(f"[{name}] 이미 있음: {out_path}")
        return

    vectors = []  # 파일 벡터를 하나씩 담는다.
    for i, item in enumerate(items):
        wave = load_audio(item["path"])  # 16kHz 모노 → 무음 파일이면 오류 → 앞뒤 무음 자르기 → 최대 진폭 1.0
        vectors.append(file_vector(wave, extractor, model))
        if (i + 1) % 1000 == 0:  # 진행 상황
            print(f"  [{name}] {i + 1}/{len(items)}")

    vectors = np.stack(vectors).astype(np.float32)  # (파일 수, 1024)
    np.savez(out_path, vectors=vectors,  # 벡터와 함께 파일 정보를 같은 순서로 저장한다.
             file_id=[it["file_id"] for it in items], label=[it["label"] for it in items],
             attack_id=[it["attack_id"] for it in items], speaker_id=[it["speaker_id"] for it in items])
    print(f"[{name}] 저장: {vectors.shape}, 값 범위 {vectors.min():.2f} ~ {vectors.max():.2f}, 평균 {vectors.mean():.3f}")


def load_split(name):  # 저장한 벡터 묶음을 불러온다. (rdm_train.py, rdm_evaluate.py에서 쓴다)

    data = np.load(os.path.join(FEATURE_DIR, f"{name}.npz"))
    return {key: data[key] for key in data.files}  # vectors, file_id, label, attack_id, speaker_id


# ===============================
# 4. 실행
# ===============================

if __name__ == "__main__":

    os.makedirs(FEATURE_DIR, exist_ok=True)
    extractor, model = load_xlsr()
    print(f"XLS-R: {XLSR_MODEL}, {XLSR_LAYER}층, 장치 {DEVICE}")

    build_split(select_bonafide(TRAIN_DATA), "train", extractor, model)  # train real 2,580개
    build_split(select_bonafide(DEV_DATA), "dev", extractor, model)  # dev real 2,548개
    build_split(select_eval_items(), "eval", extractor, model)  # eval 8,000개 (real + fake)
