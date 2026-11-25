# ===============================
# train.py
# 학습 real 파일 벡터(XLS-R 22층 프레임 평균, 1024)만으로 작은 MLP 디퓨전(DDPM)을 학습한다.
# MLP는 "노이즈가 섞인 벡터 x_t"와 "단계 t"를 보고 "섞인 노이즈 ε"를 맞히도록 배운다.
#   입력이 그림(2D)이 아니라 숫자 1024개가 한 줄로 놓인 벡터라서 "주변 칸"을 보는 Conv가 필요 없다. Linear로 충분하다.
#   폭은 처음부터 끝까지 1024로 같다. (U-Net처럼 줄였다 늘리지 않는다. 노이즈 맞히기 자체가 베끼기를 막는다)
#
# 읽는 파일: outputs/rdm_features_xlsr_l22/ 의 TRAIN_DATA·DEV_DATA 벡터 (preprocess.py가 만든 것)
# 만드는 파일 (outputs/<RUN_NAME>/checkpoints 폴더)
#   best.pt         : dev real 손실이 가장 낮았던 모델 + 표준화 통계(train real 평균·표준편차)
#   history.csv     : VAL_EVERY걸음마다 train/dev 손실
#   loss_curve.png  : 손실 곡선
#
# 이 파일의 forward_diffusion, standardize, RDM은 evaluate.py와 웹(explain.py)에서도 그대로 쓴다.
# ===============================

import csv  # history.csv를 쓴다.
import math  # timestep embedding에서 log를 계산한다.
import os  # 파일 경로를 다룬다.
import time  # 걸린 시간을 잰다.

import matplotlib  # 그림을 저장한다.
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt  # 손실 곡선을 그린다.
import torch  # 텐서 계산과 학습
import torch.nn as nn  # 신경망 층(Linear, LayerNorm 등)

from config import *  # config.py의 모든 설정값을 가져온다.
from preprocess import load_many  # 저장한 파일 벡터를 불러온다.


# ===============================
# 1. 노이즈 스케줄 (미리 한 번만 계산해 둔다)
# ===============================

betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device=DEVICE)  # 단계마다 추가할 노이즈 크기 β_t (선형으로 증가)
alphas = 1.0 - betas  # α_t = 1 − β_t : 단계마다 원본이 남는 비율
alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t = α_1 × α_2 × … × α_t : t단계까지 원본이 남는 비율
sqrt_alpha_bars = torch.sqrt(alpha_bars)  # √ᾱ_t : x_t에서 원본 x₀에 곱하는 값
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)  # √(1−ᾱ_t) : x_t에서 노이즈 ε에 곱하는 값


# ===============================
# 2. Forward diffusion (원본에 노이즈 섞기)
# ===============================

def select_random_timesteps(batch_size, generator=None):  # 벡터마다 t를 0 ~ TIMESTEPS-1 중 하나씩 무작위로 뽑는다.

    t = torch.randint(0, TIMESTEPS, (batch_size,), generator=generator)  # generator를 주면 그 난수로 뽑는다. (검증용)

    return t.to(DEVICE)  # (B,) 정수


def forward_diffusion(x0, t, noise=None):  # x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε     x0: (B, 1024), t: (B,)

    if noise is None:  # 노이즈를 밖에서 주지 않았으면 (학습할 때)
        noise = torch.randn_like(x0)  # 표준정규분포 노이즈를 새로 만든다.
    # 검증·평가 때는 고정 노이즈를 밖에서 넣어 매번 같은 문제를 풀게 한다.

    sqrt_ab = sqrt_alpha_bars[t][:, None]  # (B,) → (B, 1): 벡터마다 다른 t의 값을 1024칸 전체에 곱하도록
    sqrt_1m_ab = sqrt_one_minus_alpha_bars[t][:, None]  # (B, 1)

    x_t = sqrt_ab * x0 + sqrt_1m_ab * noise  # 노이즈가 섞인 벡터

    return x_t, noise  # MLP 입력과 정답 노이즈


# ===============================
# 3. Timestep embedding (숫자 t → 벡터)
# ===============================

def get_timestep_embedding(t, dim):  # t를 sin·cos 값 dim개로 바꾼다. (Transformer의 위치 인코딩과 같은 방식)

    half = dim // 2  # 절반은 sin, 절반은 cos
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)  # 빠른 진동부터 느린 진동까지 half개의 주파수
    args = t.float()[:, None] * freqs[None, :]  # (B, half): t마다 주파수를 곱한다.
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)

    return emb  # 가까운 t는 비슷한 벡터, 먼 t는 다른 벡터가 된다.


# ===============================
# 4. 표준화 (train real 통계)
# ===============================

def compute_stats(vectors):  # 차원마다 평균·표준편차 (train real만으로)

    mean = vectors.mean(axis=0)  # (1024,)
    std = vectors.std(axis=0) + 1e-6  # (1024,): 0으로 나누지 않도록 아주 작은 값을 더한다.

    return mean, std


def standardize(vectors, mean, std):  # 차원마다 평균 0, 표준편차 1 근처로 맞춘다. (22층 값은 표준편차 약 16 → 디퓨전 노이즈 ε(표준편차 1)와 크기를 맞추려고)

    return (vectors - mean) / std


# ===============================
# 5. MLP (노이즈를 맞히는 선수)
# ===============================

class MLPBlock(nn.Module):  # LayerNorm → Linear → (t 더함) → SiLU → Dropout → Linear, 그리고 입력을 더하는 skip

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN)  # 값의 크기를 일정하게 맞춰 학습을 안정시킨다.
        self.fc1 = nn.Linear(HIDDEN, HIDDEN)  # 1024 → 1024
        self.time_proj = nn.Linear(TIME_EMBEDDING_DIM, HIDDEN)  # t 벡터(128)를 이 블록의 크기(1024)에 맞게 바꾼다.
        self.act = nn.SiLU()  # 활성화 함수
        self.drop = nn.Dropout(DROPOUT)  # 학습 때만 일부 값을 무작위로 0으로 만든다. (외우지 않도록)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)  # 한 번 더 Linear

    def forward(self, h, t_emb):  # h: (B, HIDDEN), t_emb: (B, TIME_EMBEDDING_DIM)

        z = self.fc1(self.norm(h))  # 정규화 → Linear
        z = self.act(z + self.time_proj(t_emb))  # t 정보를 더하고 활성화 (지금 노이즈가 얼마나 섞였는지 알려 준다)
        z = self.fc2(self.drop(z))  # Dropout → Linear

        return h + z  # skip: 입력을 더해 정보가 잘 흐르게 한다.


class RDM(nn.Module):  # 노이즈 섞인 벡터 x_t (B, 1024)와 t (B,) → 예측 노이즈 (B, 1024)

    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(  # sin·cos 벡터를 학습 가능한 층에 한 번 통과시킨다.
            nn.Linear(TIME_EMBEDDING_DIM, TIME_EMBEDDING_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBEDDING_DIM, TIME_EMBEDDING_DIM),
        )
        self.inp = nn.Linear(XLSR_DIM, HIDDEN)  # 입력(1024) → 안쪽 공간(1024)
        self.blocks = nn.ModuleList([MLPBlock() for _ in range(BLOCKS)])  # 블록 3개
        self.out = nn.Sequential(  # 안쪽 공간 → 노이즈 예측 (1024)
            nn.LayerNorm(HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, XLSR_DIM),
        )

    def forward(self, x, t):

        t_emb = self.time_mlp(get_timestep_embedding(t, TIME_EMBEDDING_DIM))  # (B,) → (B, 128)
        h = self.inp(x)  # (B, 1024)
        for block in self.blocks:
            h = block(h, t_emb)  # 블록마다 t 정보를 다시 넣는다.

        return self.out(h)  # (B, 1024): 예측한 ε


# ===============================
# 6. 검증 (dev real, 고정 t·노이즈)
# ===============================

def make_val_problem(num):  # dev real 벡터마다 t 하나와 노이즈 하나를 고정한다. (매번 같은 문제를 풀게 해서 손실을 비교할 수 있게)

    generator = torch.Generator().manual_seed(VAL_SEED)  # 학습 난수와 섞지 않는 전용 난수
    t = select_random_timesteps(num, generator=generator)  # (dev 수,)
    noise = torch.randn((num, XLSR_DIM), generator=generator).to(DEVICE)  # (dev 수, 1024)

    return t, noise


def validate(model, x_dev, val_t, val_noise):  # dev real 평균 MSE

    model.eval()  # Dropout을 끈다.
    with torch.no_grad():
        x_t, _ = forward_diffusion(x_dev, val_t, val_noise)  # 고정 노이즈를 섞는다.
        loss = ((model(x_t, val_t) - val_noise) ** 2).mean()  # (예측 − 정답)²의 평균
    model.train()  # 다시 학습 모드로

    return loss.item()


# ===============================
# 7. 저장
# ===============================

def save_loss_curve(history, out_path):  # train/dev 손실 곡선

    steps = [row["step"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, [row["train_loss"] for row in history], label="train")
    ax.plot(steps, [row["val_loss"] for row in history], label="val (dev real)")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_title(f"{RUN_NAME}: XLS-R layer {XLSR_LAYER} file vectors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ===============================
# 8. 실행
# ===============================

if __name__ == "__main__":

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ----- 데이터: train real로 표준화 통계를 정하고 train·dev에 똑같이 적용 -----
    variants = ("clean", "aug") if AUGMENT else ("clean",)  # 증강을 쓰면 원본과 증강본을 함께
    train_vectors = load_many(TRAIN_DATA, variants)["vectors"]  # (학습 real 수, 1024)
    dev_vectors = load_many(DEV_DATA)["vectors"]  # (dev real 수, 1024), 원본 그대로
    mean, std = compute_stats(train_vectors)
    x_train = torch.tensor(standardize(train_vectors, mean, std), device=DEVICE)  # 벡터가 작아서 전부 GPU에 올려 둔다.
    x_dev = torch.tensor(standardize(dev_vectors, mean, std), device=DEVICE)
    print(f"학습 {TRAIN_DATA} × {variants}: {tuple(x_train.shape)}, dev {DEV_DATA}: {tuple(x_dev.shape)}")
    print(f"표준화 뒤 train 값 범위 {x_train.min():.1f} ~ {x_train.max():.1f}")

    # ----- 모델 -----
    model = RDM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"RDM 가중치 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    val_t, val_noise = make_val_problem(len(x_dev))

    # ----- 학습: STEPS걸음. 학습 real을 섞어서 앞에서부터 256개씩, 다 쓰면 다시 섞는다 -----
    history, best_loss, start = [], float("inf"), time.time()
    order, pos = torch.randperm(len(x_train), device=DEVICE), 0  # 순서와 지금 위치
    for step in range(1, STEPS + 1):
        if pos + BATCH_SIZE > len(x_train):  # 한 바퀴(epoch)를 다 돌면 순서를 다시 섞는다.
            order, pos = torch.randperm(len(x_train), device=DEVICE), 0
        idx = order[pos:pos + BATCH_SIZE]  # 이번 묶음 256개
        pos += BATCH_SIZE

        x0 = x_train[idx]  # (B, 1024)
        t = select_random_timesteps(len(idx))  # 벡터마다 무작위 t
        x_t, noise = forward_diffusion(x0, t)  # 노이즈를 섞는다.
        loss = ((model(x_t, t) - noise) ** 2).mean()  # 섞인 노이즈를 얼마나 맞혔나 (MSE)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # gradient가 갑자기 커지는 것을 막는다.
        optimizer.step()

        if step % VAL_EVERY == 0:  # 500걸음마다 dev real 손실
            val_loss = validate(model, x_dev, val_t, val_noise)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            if val_loss < best_loss:  # 가장 좋았던 모델만 남긴다. (조기 종료와 같은 효과)
                best_loss = val_loss
                torch.save({"step": step, "val_loss": val_loss, "layer": XLSR_LAYER,
                            "train_data": TRAIN_DATA, "augment": AUGMENT,  # 무엇으로 학습했는지
                            "mean": mean, "std": std,  # 평가·웹에서 똑같이 표준화하려고 함께 저장한다.
                            "model": model.state_dict()}, os.path.join(CHECKPOINT_DIR, "best.pt"))
            if step % (VAL_EVERY * 10) == 0:  # 5,000걸음마다 출력
                print(f"step {step:5d} | train {loss.item():.4f} | dev real {val_loss:.4f} | best {best_loss:.4f} | {time.time() - start:.0f}초")

    # ----- 기록 -----
    with open(os.path.join(CHECKPOINT_DIR, "history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    save_loss_curve(history, os.path.join(CHECKPOINT_DIR, "loss_curve.png"))
    print(f"끝: best dev real 손실 {best_loss:.4f}, 저장 {CHECKPOINT_DIR}")
