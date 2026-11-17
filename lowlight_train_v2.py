"""노이즈 억제 신경망(NoiseSuppressor) 학습 — 2단계 학습.

1단계(이미 끝남): 원본 Zero-DCE 사전학습 가중치(snapshots/Epoch99.pth)  -> 여기서는 얼리고(freeze) 그대로 사용
2단계(이 스크립트): 그 뒤에 붙은 작은 CNN만 학습

정답(깨끗한 이미지)이 없는 Zero-DCE 설정에서 노이즈 억제망을 학습시키기 위해 "합성 노이즈"를 쓴다.
  1) 학습 이미지 x에 센서 노이즈(읽기 노이즈 + 신호 의존 노이즈)를 더해 x_noisy를 만든다.
  2) 얼려둔 DCE-Net으로 e_clean = DCE(x), e_noisy = DCE(x_noisy)를 만든다.
     (곡선이 노이즈를 같이 증폭하기 때문에 e_noisy에는 "밝기 증폭으로 커진 노이즈"가 자연스럽게 생긴다)
  3) 신경망이 e_noisy(+gain 맵)로부터 노이즈 마스크 M과 잔차 R을 예측해 out = e_noisy - M*R 을 만든다.
  4) 손실:  L1(out, e_clean)  +  lam_mask * MSE(M, 실제 노이즈 크기 맵)
     (선택) --lam_edge: 경계 픽셀의 L1 가중치를 키움, --lam_grad: 출력/정답의 Sobel 기울기 차이 손실 추가 -> 경계 흐림 방지
     - 일부 배치(p_clean)는 노이즈를 넣지 않아서, 노이즈가 없는 곳은 건드리지 않도록 같이 학습한다.
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
import torch.optim

import dataloader
import model_v2


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_synthetic_noise(x, p_clean, read_range, shot_max):
    """x(0~1)에 이미지마다 다른 세기의 센서 노이즈를 더한다.  분산 = a*x + sigma_read^2"""
    b = x.size(0)
    dev = x.device
    sigma_read = torch.empty(b, 1, 1, 1, device=dev).uniform_(*read_range)
    a_shot = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.0, shot_max)
    keep_clean = (torch.rand(b, 1, 1, 1, device=dev) < p_clean).float()
    std = (a_shot * x + sigma_read ** 2).sqrt() * (1.0 - keep_clean)
    return (x + torch.randn_like(x) * std).clamp(0.0, 1.0)


def mask_target(e_noisy, e_clean, scale, pool=5):
    """밝아진 뒤 실제로 노이즈가 얼마나 생겼는지(|e_noisy - e_clean|)를 0~1 맵으로 만든다."""
    d = (e_noisy - e_clean).abs().mean(1, keepdim=True)
    d = F.avg_pool2d(d, pool, 1, pool // 2, count_include_pad=False)
    return (d * scale).clamp(0.0, 1.0)


def luma(x):
    """RGB -> 밝기(휘도). 경계는 대부분 밝기에 있다."""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def sobel_xy(y):
    """Sobel 기울기 (gx, gy). 8로 나눠서 '이웃 픽셀 간 밝기 차이' 정도의 크기로 맞춘다."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3) / 8.0
    p = F.pad(y, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(p, kx), F.conv2d(p, kx.transpose(2, 3))


def edge_weight_map(target, ref=0.05):
    """정답 이미지에서 경계를 찾아 0(평평) ~ 1(뚜렷한 경계) 맵으로 만든다. (N,1,H,W)
    ref=0.05는 이웃 픽셀 간 밝기 차이가 약 0.1 이상인 경계를 1로 본다."""
    y = F.avg_pool2d(luma(target), 3, 1, 1, count_include_pad=False)
    gx, gy = sobel_xy(y)
    return ((gx ** 2 + gy ** 2).sqrt() / ref).clamp(0.0, 1.0)


def recon_loss(out, target, lam_edge=0.0, lam_grad=0.0):
    """복원 손실 = (경계 가중) L1 + lam_grad * 기울기 손실.
    lam_edge: 경계 픽셀의 L1 가중치를 (1 + lam_edge)배로 키운다 -> 경계를 깎으면 손해가 커진다.
    lam_grad: 출력과 정답의 Sobel 기울기 차이(L1) -> 경계의 날카로움 자체를 맞춘다.
    둘 다 0이면 기존과 같은 단순 L1이다."""
    d = (out - target).abs()
    if lam_edge > 0:
        w = 1.0 + lam_edge * edge_weight_map(target)
        l_rec = (w * d).sum() / w.expand_as(d).sum()
    else:
        l_rec = d.mean()
    l_grad = out.new_zeros(())
    if lam_grad > 0:
        gxo, gyo = sobel_xy(luma(out))
        gxt, gyt = sobel_xy(luma(target))
        l_grad = (gxo - gxt).abs().mean() + (gyo - gyt).abs().mean()
    return l_rec + lam_grad * l_grad, l_rec, l_grad


def train(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = pick_device()
    print("device:", device)

    net = model_v2.enhance_net_denoise(ch=config.channels).to(device)
    net.load_dce(config.dce_weights, map_location=device)
    net.freeze_dce()
    n_params = sum(p.numel() for p in net.denoiser.parameters())
    print("denoiser params:", n_params)

    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.Adam(net.denoiser.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    os.makedirs(config.snapshots_folder, exist_ok=True)
    read_range = (config.read_min, config.read_max)

    net.denoiser.train()
    for epoch in range(config.num_epochs):
        for it, img in enumerate(train_loader):
            img = img.to(device)
            if config.crop_size > 0:  # 작은 조각으로 학습 (CPU 등 느린 환경에서 빠르게 확인할 때)
                cs = config.crop_size
                i = random.randint(0, img.size(2) - cs)
                j = random.randint(0, img.size(3) - cs)
                img = img[:, :, i:i + cs, j:j + cs]
            with torch.no_grad():
                x_noisy = add_synthetic_noise(img, config.p_clean, read_range, config.shot_max)
                _, e_clean, _ = net.dce(img)
                _, e_noisy, _ = net.dce(x_noisy)
                m_target = mask_target(e_noisy, e_clean, config.mask_scale)

            out, m, _ = net.denoise(e_noisy, x_noisy)
            loss_rec_total, loss_rec, loss_grad = recon_loss(out, e_clean, config.lam_edge, config.lam_grad)
            loss_mask = F.mse_loss(m, m_target)
            loss = loss_rec_total + config.lam_mask * loss_mask

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.denoiser.parameters(), config.grad_clip_norm)
            optimizer.step()

            if (it + 1) % config.display_iter == 0:
                with torch.no_grad():
                    base = F.l1_loss(e_noisy, e_clean).item()
                print("epoch %d iter %d | loss %.4f (rec %.4f, grad %.4f, mask %.4f) | L1 before %.4f -> after %.4f"
                      % (epoch, it + 1, loss.item(), loss_rec.item(), loss_grad.item(), loss_mask.item(),
                         base, F.l1_loss(out, e_clean).item()))

        torch.save(net.denoiser.state_dict(),
                   os.path.join(config.snapshots_folder, "denoiser_epoch%d.pth" % epoch))
        torch.save(net.denoiser.state_dict(),
                   os.path.join(config.snapshots_folder, "denoiser_latest.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowlight_images_path", type=str, default="data/train_data/")
    parser.add_argument("--dce_weights", type=str, default="snapshots/Epoch99.pth")
    parser.add_argument("--snapshots_folder", type=str, default="snapshots/denoise_v1/")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--display_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1143)
    parser.add_argument("--crop_size", type=int, default=0, help="0이면 256x256 전체 사용, 값을 주면 그 크기로 랜덤 크롭")
    # 합성 노이즈 설정 (이미지 범위 0~1 기준 표준편차)
    parser.add_argument("--read_min", type=float, default=0.005, help="읽기 노이즈 최소 std")
    parser.add_argument("--read_max", type=float, default=0.03, help="읽기 노이즈 최대 std")
    parser.add_argument("--shot_max", type=float, default=0.01, help="신호 의존 노이즈 계수 최대값")
    parser.add_argument("--p_clean", type=float, default=0.2, help="노이즈를 넣지 않는 이미지 비율")
    # 손실 설정
    parser.add_argument("--lam_mask", type=float, default=0.1)
    parser.add_argument("--lam_edge", type=float, default=0.0,
                        help="경계 픽셀 L1 가중치 추가분 (0=사용 안 함, 예: 4면 경계에서 5배)")
    parser.add_argument("--lam_grad", type=float, default=0.0,
                        help="기울기 손실 가중치 (0=사용 안 함). 2026-11-17 실험에서는 오히려 경계를 흐리게 해서 비권장")
    parser.add_argument("--mask_scale", type=float, default=15.0, help="노이즈 크기 -> 마스크 정답 변환 배율")
    train(parser.parse_args())
