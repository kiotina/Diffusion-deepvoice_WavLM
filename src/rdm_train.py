# ===============================
# rdm_train.py
# train real 파일 벡터(XLS-R 22층 프레임 평균, 1024)만으로 작은 MLP 디퓨전을 학습한다.
# 디퓨전 부분(노이즈 일정, 노이즈 섞기, t 임베딩, ε 예측, MSE)은 train.py와 똑같다. 노이즈를 맞히는 선수만 U-Net → MLP로 바뀐다.
#   입력이 그림(2D)이 아니라 숫자 1024개가 한 줄로 놓인 벡터라서 "주변 칸"을 보는 Conv가 필요 없다. Linear로 충분하다.
#
# 읽는 파일: outputs/rdm_xlsr_l22/features/train.npz, dev.npz (rdm_features.py가 만든 것)
# 만드는 파일 (outputs/rdm_xlsr_l22/checkpoints 폴더)
#   best.pt         : dev real 손실이 가장 낮았던 모델 + 표준화 통계(train real 평균·표준편차)
#   history.csv     : RDM_VAL_EVERY epoch마다 train/dev 손실
#   loss_curve.png  : 손실 곡선
# ===============================

import csv  # history.csv를 쓴다.
import os  # 파일 경로를 다룬다.
import time  # 걸린 시간을 잰다.

import matplotlib  # 그림을 저장한다.
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다.
import matplotlib.pyplot as plt  # 손실 곡선을 그린다.
import numpy as np  # 통계를 계산한다.
import torch  # 텐서 계산과 학습
import torch.nn as nn  # 신경망 층(Linear, LayerNorm 등)

from config import *  # config.py의 모든 설정값을 가져온다.
from rdm_features import load_split  # 저장한 파일 벡터를 불러온다.
from train import forward_diffusion, get_timestep_embedding, select_random_timesteps  # U-Net 학습과 같은 디퓨전 부품

CHECKPOINT_DIR_RDM = os.path.join(RDM_DIR, "checkpoints")  # 학습한 모델을 저장할 폴더


# ===============================
# 1. 표준화 (train real 통계)
# ===============================

def compute_stats(vectors):  # 차원마다 평균·표준편차 (train real만으로)

    mean = vectors.mean(axis=0)  # (1024,)
    std = vectors.std(axis=0) + 1e-6  # (1024,): 0으로 나누지 않도록 아주 작은 값을 더한다.

    return mean, std


def standardize(vectors, mean, std):  # 차원마다 평균 0, 표준편차 1 근처로 맞춘다. (디퓨전 노이즈 ε와 크기를 맞추려고)

    return (vectors - mean) / std


# ===============================
# 2. MLP (노이즈를 맞히는 선수)
# ===============================

class MLPBlock(nn.Module):  # LayerNorm → Linear → SiLU → (t 더함) → Dropout → Linear, 그리고 입력을 더하는 skip (train.py ResBlock과 같은 생각)

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(RDM_HIDDEN)  # 값의 크기를 일정하게 맞춰 학습을 안정시킨다. (U-Net의 GroupNorm 역할)
        self.fc1 = nn.Linear(RDM_HIDDEN, RDM_HIDDEN)  # 1024 → 1024
        self.time_proj = nn.Linear(TIME_EMBEDDING_DIM, RDM_HIDDEN)  # t 벡터를 이 블록의 크기에 맞게 바꾼다.
        self.act = nn.SiLU()  # 활성화 함수
        self.drop = nn.Dropout(RDM_DROPOUT)  # 학습 때만 일부 값을 무작위로 0으로 만든다. (외우지 않도록)
        self.fc2 = nn.Linear(RDM_HIDDEN, RDM_HIDDEN)  # 한 번 더 Linear

    def forward(self, h, t_emb):  # h: (B, RDM_HIDDEN), t_emb: (B, TIME_EMBEDDING_DIM)

        z = self.fc1(self.norm(h))  # 정규화 → Linear
        z = self.act(z + self.time_proj(t_emb))  # t 정보를 더하고 활성화 (지금 노이즈가 얼마나 섞였는지 알려 준다)
        z = self.fc2(self.drop(z))  # Dropout → Linear

        return h + z  # skip: 입력을 더해 정보가 잘 흐르게 한다.


class RDM(nn.Module):  # 노이즈 섞인 벡터 x_t (B, 1024)와 t (B,) → 예측 노이즈 (B, 1024)

    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(  # sin·cos 벡터를 학습 가능한 층에 한 번 통과시킨다. (U-Net과 같다)
            nn.Linear(TIME_EMBEDDING_DIM, TIME_EMBEDDING_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBEDDING_DIM, TIME_EMBEDDING_DIM),
        )
        self.inp = nn.Linear(XLSR_DIM, RDM_HIDDEN)  # 1024 → 1024 (입력을 안쪽 공간으로)
        self.blocks = nn.ModuleList([MLPBlock() for _ in range(RDM_BLOCKS)])  # 블록 3개
        self.out = nn.Sequential(  # 안쪽 공간 → 노이즈 예측 (1024)
            nn.LayerNorm(RDM_HIDDEN),
            nn.SiLU(),
            nn.Linear(RDM_HIDDEN, XLSR_DIM),
        )

    def forward(self, x, t):

        t_emb = self.time_mlp(get_timestep_embedding(t, TIME_EMBEDDING_DIM))  # (B,) → (B, 128)
        h = self.inp(x)  # (B, 1024)
        for block in self.blocks:
            h = block(h, t_emb)  # 블록마다 t 정보를 다시 넣는다.

        return self.out(h)  # (B, 1024): 예측한 ε


# ===============================
# 3. 검증 (dev real, 고정 t·노이즈)
# ===============================

def make_val_problem(num):  # dev real 벡터마다 t 하나와 노이즈 하나를 고정한다. (epoch마다 같은 문제를 풀게 해서 손실을 비교할 수 있게)

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
# 4. 저장
# ===============================

def save_loss_curve(history, out_path):  # train/dev 손실 곡선

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs, [row["val_loss"] for row in history], label="val (dev real)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"{RDM_RUN_NAME}: XLS-R layer {XLSR_LAYER} file vectors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ===============================
# 5. 실행
# ===============================

if __name__ == "__main__":

    os.makedirs(CHECKPOINT_DIR_RDM, exist_ok=True)

    # ----- 데이터: train real로 표준화 통계를 정하고 train·dev에 똑같이 적용 -----
    train_vectors = load_split("train")["vectors"]  # (2580, 1024)
    dev_vectors = load_split("dev")["vectors"]  # (2548, 1024)
    mean, std = compute_stats(train_vectors)
    x_train = torch.tensor(standardize(train_vectors, mean, std), device=DEVICE)  # 벡터가 작아서 전부 GPU에 올려 둔다.
    x_dev = torch.tensor(standardize(dev_vectors, mean, std), device=DEVICE)
    print(f"train {tuple(x_train.shape)}, dev {tuple(x_dev.shape)}, 표준화 뒤 train 값 범위 {x_train.min():.1f} ~ {x_train.max():.1f}")

    # ----- 모델 -----
    model = RDM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"RDM 가중치 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    val_t, val_noise = make_val_problem(len(x_dev))

    # ----- 학습 -----
    history, best_loss, start = [], float("inf"), time.time()
    for epoch in range(1, RDM_EPOCHS + 1):
        order = torch.randperm(len(x_train), device=DEVICE)  # epoch마다 순서를 섞는다.
        for idx in order.split(RDM_BATCH_SIZE):  # 256개씩
            x0 = x_train[idx]  # (B, 1024)
            t = select_random_timesteps(len(idx))  # 벡터마다 무작위 t
            x_t, noise = forward_diffusion(x0, t)  # 노이즈를 섞는다.
            loss = ((model(x_t, t) - noise) ** 2).mean()  # 섞인 노이즈를 얼마나 맞혔나 (MSE)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # gradient가 갑자기 커지는 것을 막는다.
            optimizer.step()

        if epoch % RDM_VAL_EVERY == 0:  # 50 epoch마다 dev real 손실
            val_loss = validate(model, x_dev, val_t, val_noise)
            history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss})
            if val_loss < best_loss:  # 가장 좋았던 모델만 남긴다. (조기 종료와 같은 효과)
                best_loss = val_loss
                torch.save({"epoch": epoch, "val_loss": val_loss, "layer": XLSR_LAYER,
                            "mean": mean, "std": std,  # 평가·웹에서 똑같이 표준화하려고 함께 저장한다.
                            "model": model.state_dict()}, os.path.join(CHECKPOINT_DIR_RDM, "best.pt"))
            if epoch % (RDM_VAL_EVERY * 10) == 0:  # 500 epoch마다 출력
                print(f"epoch {epoch:4d} | train {loss.item():.4f} | dev real {val_loss:.4f} | best {best_loss:.4f} | {time.time() - start:.0f}초")

    # ----- 기록 -----
    with open(os.path.join(CHECKPOINT_DIR_RDM, "history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    save_loss_curve(history, os.path.join(CHECKPOINT_DIR_RDM, "loss_curve.png"))
    print(f"끝: best dev real 손실 {best_loss:.4f}, 저장 {CHECKPOINT_DIR_RDM}")
