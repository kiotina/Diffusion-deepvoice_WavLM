# ===============================
# server.py
# 브라우저에서 음성을 받아 explain.py로 분석하고, 결과를 JSON으로 돌려주는 웹 서버
#
# 실행 (venv 환경, 프로젝트 폴더에서)
#   python src/explain.py     ← 처음 한 번 (기준값 만들기)
#   python src/server.py      ← 서버 켜기 → 브라우저에서 http://127.0.0.1:8000
#
# 주소
#   GET  /              : web/index.html
#   GET  /api/info      : 모델 정보, 임계값, 업로드 제한
#   POST /api/analyze   : 음성 파일 원본 바이트를 받아 판정과 설명을 돌려준다.
#                         ?consent=yes&ext=mp3  (파일 이름은 받지 않는다. 웹은 파일 업로드만 한다)
#
# 개인정보 보호
#   1. 동의(consent=yes)가 없으면 음성을 처리하지 않는다.
#   2. 파일 이름은 받지 않는다. 브라우저가 확장자만 보낸다.
#   3. 음성은 메모리에서만 처리한다. ffmpeg에도 파이프로 넘긴다.
#      예외: 파이프로 읽을 수 없는 파일(일부 m4a·mp4)만 임시 파일을 만들고, 읽자마자 지운다. (결과에 표시)
#   4. 결과·음성을 저장하지 않고, 접속 기록(IP, 주소)도 남기지 않는다.
#   5. 브라우저가 결과를 캐시하지 않도록 하고, 외부 사이트로의 연결을 막는 보안 헤더를 붙인다.
# ===============================

import os  # 파일 경로
import re  # ffmpeg 출력에서 샘플레이트·채널 읽기
import secrets  # 분석 ID (무작위, 어떤 정보와도 연결되지 않음)
import subprocess  # ffmpeg 실행
import tempfile  # 일부 형식용 임시 파일
import threading  # GPU를 한 번에 한 요청만 쓰도록 잠근다.
from contextlib import asynccontextmanager  # 서버를 켤 때 모델을 한 번만 불러온다.
from datetime import datetime

import imageio_ffmpeg  # venv 안의 ffmpeg (preprocess.py 코덱 증강과 같은 것)
import librosa  # 모노 변환·리샘플링 (librosa.load와 같은 방식)
import numpy as np
import uvicorn  # 웹 서버 실행
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool  # 모델 계산을 별도 스레드에서 (서버가 멈추지 않도록)
from fastapi.responses import FileResponse, JSONResponse

from config import *

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ALLOWED_EXTS = {"wav", "flac", "mp3", "m4a", "aac", "ogg", "oga", "opus", "webm", "mp4", "3gp", "amr", "wma"}  # 받을 확장자
LAYOUT_CHANNELS = {"mono": 1, "stereo": 2, "2.1": 3, "3.0": 3, "quad": 4, "4.0": 4, "5.0": 5, "5.1": 6, "6.1": 7, "7.1": 8}  # ffmpeg 채널 이름 → 채널 수

SECURITY_HEADERS = {
    "Cache-Control": "no-store",  # 브라우저·프록시가 결과(음성에서 나온 정보)를 저장하지 않게
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",  # 다른 사이트가 이 화면을 몰래 끼워 넣지 못하게
    "Permissions-Policy": "microphone=(), camera=(), geolocation=()",  # 녹음 기능이 없으므로 마이크도 막는다.
    "Content-Security-Policy": (  # 이 서버 말고 다른 곳으로는 어떤 요청도 못 나가게 (외부 글꼴·스크립트·분석 도구 없음)
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}

explainer = None  # 서버를 켤 때 만든다.
gpu_lock = threading.Lock()  # 한 번에 한 요청만 모델을 쓴다.


# ===============================
# 1. 음성 디코딩 (메모리에서)
# ===============================

def run_ffmpeg(source, data, extra_args):  # source: "pipe:0"(메모리) 또는 임시 파일 경로

    command = [FFMPEG, "-hide_banner", "-i", source, "-map", "0:a:0", *extra_args,
               "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]  # 원래 샘플레이트·채널 그대로 32비트 실수로 푼다.
    if data is not None:
        return subprocess.run(command, input=data, capture_output=True, timeout=120)
    return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=120)


def decode_once(data, ext, extra_args):  # 먼저 파이프로, 안 되면 임시 파일로 (반환: 결과, 임시 파일을 썼는지)

    completed = run_ffmpeg("pipe:0", data, extra_args)
    if completed.returncode == 0 and completed.stdout:
        return completed, False

    # mp4·m4a처럼 파일 끝에 목차가 있는 형식은 앞에서부터 흘려 읽을 수 없다. 이때만 임시 파일을 쓴다.
    fd, path = tempfile.mkstemp(prefix="dv_", suffix="." + ext if ext else "")  # 본인만 읽을 수 있는 권한으로 만든다.
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        completed = run_ffmpeg(path, None, extra_args)
    finally:
        os.remove(path)  # 읽자마자 지운다.

    return completed, True


def stream_info(stderr, section):  # ffmpeg 출력에서 "Audio: 코덱, 44100 Hz, stereo" 부분 읽기

    match = re.search(r"Audio: ([^\s,(]+)[^\n]*?, (\d+) Hz, ([^,\n]+)", section)
    if not match:
        return None, None, None

    layout = match.group(3).strip()
    count = re.match(r"(\d+) channels", layout)
    channels = int(count.group(1)) if count else LAYOUT_CHANNELS.get(layout.split("(")[0])

    return match.group(1), int(match.group(2)), channels


def decode_audio(data, ext):  # 음성 바이트 → 16kHz 모노 waveform, 원래 형식 정보

    completed, temp_used = decode_once(data, ext, [])
    stderr = completed.stderr.decode("utf-8", "replace")
    input_part, _, output_part = stderr.partition("Output #0")
    codec, in_rate, in_channels = stream_info(stderr, input_part)
    _, out_rate, out_channels = stream_info(stderr, output_part)

    if completed.returncode != 0 or not completed.stdout or out_rate is None:
        raise ValueError("음성 파일을 읽을 수 없습니다. 손상된 파일이거나 음성이 없는 파일일 수 있습니다.")

    if out_channels is None:  # 채널 배치를 모르면 ffmpeg가 모노로 합치게 다시 푼다. (드문 경우)
        completed, temp_again = decode_once(data, ext, ["-ac", "1"])
        temp_used = temp_used or temp_again
        out_channels = 1

    pcm = np.frombuffer(completed.stdout, dtype=np.float32)
    pcm = pcm[: len(pcm) // out_channels * out_channels].reshape(-1, out_channels).T  # (채널, 샘플)
    wave = librosa.to_mono(pcm) if out_channels > 1 else pcm[0]  # librosa.load와 같은 순서: 모노 → 리샘플링
    if out_rate != SAMPLE_RATE:
        wave = librosa.resample(wave, orig_sr=out_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")

    info = {"ext": ext, "codec": codec, "sample_rate": in_rate or out_rate, "channels": in_channels or out_channels}
    return np.array(wave, dtype=np.float32), info, temp_used


def analyze_bytes(data, ext, source):  # 별도 스레드에서 실행된다.

    wave, source_info, temp_used = decode_audio(data, ext)
    source_info["source"] = source

    with gpu_lock:
        result = explainer.analyze_wave(wave, source_info)

    del wave  # 참조를 끊어 메모리에서 빨리 해제되게 한다. (Python은 메모리를 0으로 덮어쓰는 것까지는 보장하지 않는다)

    result["analysis_id"] = secrets.token_hex(6)  # 보고서 구분용. 서버에는 남지 않는다.
    result["processed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    result["privacy"] = {
        "stored": False,  # 음성·결과를 서버에 저장하지 않음
        "filename_received": False,  # 파일 이름을 받지 않음
        "temp_file_used": temp_used,  # 디코딩에 임시 파일을 썼는지 (썼다면 읽자마자 삭제)
        "access_log": False,  # 접속 기록을 남기지 않음
        "processing_location": f"{SERVER_HOST}:{SERVER_PORT}",
    }

    return result


# ===============================
# 2. 서버
# ===============================

@asynccontextmanager
async def lifespan(app):

    global explainer
    from explain import Explainer  # 모델·WavLM·torch를 여기서 불러온다.
    explainer = Explainer()  # 기준값이 없거나 맞지 않으면 여기서 멈춘다.
    card = explainer.model_card()
    print(f"모델: {card['run_name']} · {card['feature']} · {card['network']} · {card['checkpoint_epoch']} · 장치 {DEVICE}")
    print("개인정보: 음성·결과 저장 안 함, 파일 이름 받지 않음, 접속 기록 없음")
    if SERVER_HOST not in ("127.0.0.1", "localhost"):
        print("주의: 외부에서 접속할 수 있게 열었습니다. HTTPS 없이 쓰면 음성이 암호화되지 않은 채 네트워크로 오갑니다.")
    print(f"브라우저에서 http://{SERVER_HOST}:{SERVER_PORT} 를 여세요.")
    yield


app = FastAPI(title="딥보이스 탐지 웹 서비스", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)  # 자동 문서 페이지는 끈다.


@app.middleware("http")
async def add_security_headers(request, call_next):  # 모든 응답에 보안 헤더를 붙인다.

    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers[name] = value

    return response


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/info")
def api_info():
    return explainer.info()


@app.post("/api/analyze")
async def api_analyze(request: Request, consent: str = "", source: str = "file", ext: str = ""):

    if consent != "yes":  # 동의가 없으면 음성을 읽지도 않는다.
        raise HTTPException(status_code=403, detail="개인정보 처리에 동의해야 분석할 수 있습니다.")

    ext = re.sub(r"[^a-z0-9]", "", ext.lower())[:5]
    if ext and ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=415, detail=f"지원하지 않는 형식입니다: {ext}")
    source = "file"  # 녹음 기능을 뺐으므로 항상 파일 업로드

    limit = MAX_UPLOAD_MB * 1024 * 1024
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다. {MAX_UPLOAD_MB}MB 이하로 올려 주세요.")

    body = bytearray()  # 디스크에 쓰지 않고 메모리로만 받는다.
    try:
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다. {MAX_UPLOAD_MB}MB 이하로 올려 주세요.")
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        result = await run_in_threadpool(analyze_bytes, bytes(body), ext, source)
        return JSONResponse(result)

    except HTTPException:
        raise
    except ValueError as error:  # 읽을 수 없는 파일, 너무 짧은 파일 등
        raise HTTPException(status_code=400, detail=str(error))
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=400, detail="음성 파일을 읽는 데 너무 오래 걸립니다.")
    except Exception as error:  # 그 밖의 오류: 서버 창에는 종류만 남기고(음성 내용·파일 정보는 남기지 않음) 화면에는 짧게
        print(f"분석 오류: {type(error).__name__}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 났습니다: {type(error).__name__}")
    finally:
        body[:] = b"\x00" * len(body)  # 받은 바이트를 0으로 덮어쓴다. (사본까지 지우지는 못하지만 남는 흔적을 줄인다)
        body.clear()


# ===============================
# 3. 실행
# ===============================

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, access_log=False, server_header=False)  # 접속 기록(IP, 주소)을 남기지 않는다.
