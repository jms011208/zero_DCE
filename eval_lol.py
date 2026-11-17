"""LOL eval15(저조도/정답 쌍)로 기존 Zero-DCE와 노이즈 억제망을 붙인 모델을 같은 조건에서 비교한다.

예)  python eval_lol.py --lol_dir data/LOL/eval15 --denoise_weights snapshots/denoise_v1/denoiser_latest.pth

출력: 이미지별/평균 PSNR, SSIM (기존 Zero-DCE vs 노이즈 억제 추가).
참고: Zero-DCE는 정답 없이 학습한 모델이라 LOL의 정답 밝기와 정확히 맞지 않아 절대 점수는 낮다.
      같은 조건에서의 "차이"를 보는 용도이다.
"""
import argparse
import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from lowlight_test_v2 import build_net, list_images, load_image, pick_device


def to_np(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def main(config):
    device = pick_device()
    net = build_net(config.dce_weights, config.denoise_weights, device, config.channels)
    low_dir = os.path.join(config.lol_dir, "low")
    high_dir = os.path.join(config.lol_dir, "high")

    rows = []
    with torch.no_grad():
        for low_path in list_images(low_dir):
            name = os.path.basename(low_path)
            high_path = os.path.join(high_dir, name)
            if not os.path.exists(high_path):
                continue
            x = load_image(low_path, device)
            gt = to_np(load_image(high_path, device))
            e, out, _, _ = net(x)
            e, out = to_np(e), to_np(out)
            rows.append((
                name,
                peak_signal_noise_ratio(gt, e, data_range=1.0),
                structural_similarity(gt, e, channel_axis=2, data_range=1.0),
                peak_signal_noise_ratio(gt, out, data_range=1.0),
                structural_similarity(gt, out, channel_axis=2, data_range=1.0),
            ))

    print("%-12s %9s %9s | %9s %9s" % ("image", "PSNR base", "SSIM base", "PSNR +dn", "SSIM +dn"))
    for r in rows:
        print("%-12s %9.3f %9.4f | %9.3f %9.4f" % r)
    arr = np.array([r[1:] for r in rows])
    mean = arr.mean(0)
    print("%-12s %9.3f %9.4f | %9.3f %9.4f" % (("mean(%d)" % len(rows),) + tuple(mean)))
    print("PSNR diff: %+.3f dB | SSIM diff: %+.4f" % (mean[2] - mean[0], mean[3] - mean[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lol_dir", type=str, default="data/LOL/eval15")
    parser.add_argument("--dce_weights", type=str, default="snapshots/Epoch99.pth")
    parser.add_argument("--denoise_weights", type=str, default="snapshots/denoise_v1/denoiser_latest.pth")
    parser.add_argument("--channels", type=int, default=16)
    main(parser.parse_args())
