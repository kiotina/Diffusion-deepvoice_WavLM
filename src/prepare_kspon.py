# ===============================
# prepare_kspon.py
# AIHub "한국어 음성"(KsponSpeech, 약 2,000명의 자유 대화 1,000시간)에서 일부만 골라 wav로 꺼낸다.
#
# 왜 필요한가: 한국어 진짜 음성 무리가 화자 4명(AIHub 다화자 음성합성 일부)뿐이라,
#              처음 듣는 사람의 진짜 목소리를 fake로 본다. (내 목소리 녹음: 거리 4.67 > 임계값 3.50)
#              → 화자가 많은 KsponSpeech를 진짜 음성 무리에 더한다. (학습 없음, feature_maha.py가 쓴다)
#
# 원본 형식: zip 안에 헤더 없는 PCM (16kHz, 16비트 정수, little endian). 대화 세션마다 폴더 하나(약 1,000문장)
# 고르는 방법 (fake는 전혀 없다. 모두 진짜 사람 음성)
#   - train: KsponSpeech_01.zip의 폴더마다 KSPON_EVERY번째 파일만 → 폴더(세션)가 고르게 섞여 화자가 많아진다.
#   - dev:   KsponSpeech_eval.zip(eval_clean, eval_other)에서 KSPON_DEV_EVERY번째 파일만 → 임계값용
#   - 둘 다 KSPON_MIN_SEC보다 짧은 문장은 버린다. (짧으면 특징 평균이 불안정하다)
#   - 화자 이름표가 없어서 화자 기준 분할은 못 한다. train(01)과 dev(eval)는 녹음 묶음이 달라 겹침이 적다고 보고 이렇게 나눈다.
# 실행: 프로젝트 맨 위 폴더에서  python src/prepare_kspon.py   (zip을 먼저 KSPON_ZIP_DIR에 만들어 둔다)
# ===============================

import os  # 경로
import zipfile  # zip을 풀지 않고 필요한 파일만 읽는다. (15GB를 다 풀면 공간·시간이 많이 든다)

import numpy as np  # PCM → 숫자
import soundfile as sf  # wav로 저장

from config import *  # KSPON_* 설정, SAMPLE_RATE


def pcm_to_wave(data):  # 헤더 없는 16비트 PCM 바이트 → -1~1 실수 waveform

    data = data[: len(data) // 2 * 2]  # 몇몇 파일은 바이트 수가 홀수라 끝의 1바이트를 버린다. (16비트 = 2바이트씩)
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0  # little endian 16비트 정수 → 실수


def extract(zip_path, out_dir, every, folders=None):  # zip 안의 PCM을 every번째마다 골라 out_dir/<폴더>/<이름>.wav로 저장

    kept, skipped = 0, 0
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".pcm"))  # 이름순 → 언제 실행해도 같은 파일을 고른다.
        groups = {}  # 폴더 이름 → 그 폴더의 PCM 목록
        for n in names:
            groups.setdefault(os.path.basename(os.path.dirname(n)), []).append(n)
        print(f"{os.path.basename(zip_path)}: PCM {len(names)}개, 폴더 {len(groups)}개")

        for folder, files in groups.items():
            if folders is not None and not any(key in folder for key in folders):
                continue
            for n in files[::every]:  # every번째마다 하나
                wave = pcm_to_wave(z.read(n))
                if len(wave) < KSPON_MIN_SEC * SAMPLE_RATE:  # 너무 짧은 문장은 버린다.
                    skipped += 1
                    continue
                path = os.path.join(out_dir, folder, os.path.splitext(os.path.basename(n))[0] + ".wav")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                sf.write(path, wave, SAMPLE_RATE)  # 16kHz wav로 저장 (load_audio가 그대로 읽는다)
                kept += 1
    print(f"  저장 {kept}개, 짧아서 버림 {skipped}개 → {out_dir}")


if __name__ == "__main__":
    extract(os.path.join(KSPON_ZIP_DIR, "KsponSpeech_01.zip"), os.path.join(KSPON_WAV_DIR, "train"), KSPON_EVERY)
    extract(os.path.join(KSPON_ZIP_DIR, "KsponSpeech_eval.zip"), os.path.join(KSPON_WAV_DIR, "dev"), KSPON_DEV_EVERY)
