"""Zero-DCE 결과에 대한 노이즈 제거 방법 비교 (LOL eval15).

같은 Zero-DCE 결과(E)에 아래 방법을 각각 적용하고, 같은 지표로 비교한다.
  base       : 노이즈 제거 없음 (기존 Zero-DCE, 기준)
  cnn        : 노이즈 억제 소형 신경망 (model_v2, --cnn_weights 필요)
  nlm        : 비지역 평균 (OpenCV fastNlMeansDenoisingColored)
  bilateral  : 양방향 필터 (OpenCV bilateralFilter)
  guided     : 가이디드 필터 (자체 구현, 채널별, E의 밝기를 안내 이미지로 사용)
  bm3d       : BM3D (pip install bm3d, 느림)
  ycc_<필터> : 밝기/색 분리. 위 필터(nlm, bilateral, guided, bm3d)를 적용한 뒤
               색(Cb, Cr)은 필터 결과를, 밝기(Y)는 기존 Zero-DCE 결과를 사용한다.
               경계는 대부분 밝기에 있으므로 경계를 지키면서 색 노이즈만 지운다.
               --ycc_luma 로 밝기에도 필터 결과를 일부 섞을 수 있다 (0=밝기 그대로, 1=일반 필터와 같음).
               예) ycc_nlm, ycc_bilateral, ycc_guided, ycc_bm3d

지표 (모두 정답 이미지 기준)
  PSNR, SSIM       : 높을수록 정답에 가까움 (밝기 차이의 영향이 큼)
  edge_ratio       : 정답의 경계 픽셀에서 '출력 기울기 / 기존 Zero-DCE 기울기'. 1이면 경계 유지, 작을수록 흐려짐
  flat_luma        : 정답이 평평한 픽셀에서 밝기 고주파 성분의 표준편차 (x1000, 낮을수록 노이즈 적음)
  flat_chroma      : 같은 위치의 색(Cb+Cr) 고주파 성분의 표준편차 (x1000, 낮을수록 색 노이즈 적음)
  sec              : 이미지 한 장당 노이즈 제거 단계 시간 (Zero-DCE 시간 제외)

예)
  python compare_denoise.py --cnn_weights snapshots/denoise_v1/denoiser_latest.pth
  python compare_denoise.py --methods base,nlm,bilateral,guided --adaptive --save_dir data/result_compare
  python compare_denoise.py --methods base,guided,ycc_guided,ycc_nlm --ycc_luma 0.3
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import model_v2
from lowlight_test_v2 import list_images, load_image, pick_device

FILTERS = ["nlm", "bilateral", "guided", "bm3d"]
ALL_METHODS = ["base", "cnn"] + FILTERS + ["ycc_" + f for f in FILTERS]
# 기본으로 실행할 방법 (ycc_bm3d는 느려서 직접 지정할 때만)
DEFAULT_METHODS = ["base", "cnn"] + FILTERS + ["ycc_nlm", "ycc_bilateral", "ycc_guided"]


# ---------------------------------------------------------------- 고정 수식 필터 (입력/출력: HxWx3 float32, 0~1)
def nlm(img, h, h_color):
    u8 = (img * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    out = cv2.fastNlMeansDenoisingColored(u8, None, h=h, hColor=h_color,
                                          templateWindowSize=7, searchWindowSize=21)
    return out.astype(np.float32) / 255.0


def bilateral(img, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(img.astype(np.float32), d, sigma_color, sigma_space)


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided(img, r, eps):
    """He et al. 가이디드 필터. 안내 이미지 I = 밝기, 각 색 채널 p를 I의 국소 선형식 a*I + b로 근사."""
    I = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    mean_I = _box(I, r)
    var_I = _box(I * I, r) - mean_I * mean_I
    out = np.empty_like(img, dtype=np.float32)
    for c in range(3):
        p = img[..., c].astype(np.float32)
        mean_p = _box(p, r)
        cov_Ip = _box(I * p, r) - mean_I * mean_p
        a = cov_Ip / (var_I + eps)          # 분산이 eps보다 큰 곳(경계)은 a~1로 그대로, 작은 곳은 평균으로
        b = mean_p - a * mean_I
        out[..., c] = _box(a, r) * I + _box(b, r)
    return out.clip(0, 1)


def bm3d_filter(img, sigma):
    import bm3d  # 필요할 때만 불러옴 (pip install bm3d)
    if sigma <= 0:  # 자동: 이미지에서 노이즈 표준편차 추정
        from skimage.restoration import estimate_sigma
        sigma = float(np.mean(estimate_sigma(img, channel_axis=-1)))
    return np.asarray(bm3d.bm3d_rgb(img, sigma), dtype=np.float32).clip(0, 1)


# ---------------------------------------------------------------- 지표
def _sobel(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(gx * gx + gy * gy)


def _luma_chroma(img):
    y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return y, 0.564 * (img[..., 2] - y), 0.713 * (img[..., 0] - y)


def ycc_combine(base, filtered, luma=0.0):
    """밝기(Y)는 base와 filtered를 luma 비율로 섞고, 색(Cb, Cr)은 filtered를 쓴다. HxWx3, 0~1."""
    yb, _, _ = _luma_chroma(base)
    yf, cbf, crf = _luma_chroma(filtered)
    y = (1.0 - luma) * yb + luma * yf
    r = y + 1.403 * crf
    g = y - 0.344 * cbf - 0.714 * crf
    b = y + 1.773 * cbf
    return np.stack([r, g, b], axis=-1).clip(0, 1).astype(np.float32)


def _highpass(ch):
    return ch - cv2.blur(ch, (5, 5), borderType=cv2.BORDER_REPLICATE)


class RegionMasks:
    """정답 이미지에서 경계/평평한 영역을 한 번만 계산해 둔다."""

    def __init__(self, gt, edge_th, flat_th):
        g = _sobel(cv2.blur(gt.mean(2).astype(np.float32), (5, 5), borderType=cv2.BORDER_REPLICATE))
        self.edge = g > edge_th
        self.flat = g < flat_th


def compute_metrics(out, base, gt, masks):
    m = {
        "psnr": peak_signal_noise_ratio(gt, out, data_range=1.0),
        "ssim": structural_similarity(gt, out, channel_axis=2, data_range=1.0),
    }
    ge_out = _sobel(out.mean(2).astype(np.float32))[masks.edge].mean()
    ge_base = _sobel(base.mean(2).astype(np.float32))[masks.edge].mean()
    m["edge_ratio"] = float(ge_out / (ge_base + 1e-8))
    y, cb, cr = _luma_chroma(out.astype(np.float32))
    m["flat_luma"] = float(_highpass(y)[masks.flat].std() * 1000)
    m["flat_chroma"] = float((_highpass(cb)[masks.flat].std() + _highpass(cr)[masks.flat].std()) * 1000)
    return m


# ---------------------------------------------------------------- 실행
def to_np(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float32)


def main(cfg):
    device = pick_device()
    methods = [m.strip() for m in cfg.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise SystemExit("알 수 없는 방법: %s (가능: %s)" % (bad, ",".join(ALL_METHODS)))
    if "cnn" in methods and not cfg.cnn_weights:
        print("[안내] --cnn_weights 가 없어 cnn은 건너뜁니다.")
        methods.remove("cnn")
    if "cnn" in methods and not os.path.exists(cfg.cnn_weights):
        print("[안내] CNN 가중치 파일이 없어 cnn은 건너뜁니다: %s" % cfg.cnn_weights)
        methods.remove("cnn")
    if any(m in ("bm3d", "ycc_bm3d") for m in methods):
        try:
            import bm3d  # noqa: F401
        except ImportError:
            print("[안내] bm3d 패키지가 없어 bm3d, ycc_bm3d는 건너뜁니다. (pip install bm3d)")
            methods = [m for m in methods if m not in ("bm3d", "ycc_bm3d")]

    net = model_v2.enhance_net_denoise(ch=cfg.channels).to(device)
    net.load_dce(cfg.dce_weights, map_location=device)
    if "cnn" in methods:
        net.denoiser.load_state_dict(torch.load(cfg.cnn_weights, map_location=device))
    net.eval()

    filters = {
        "nlm": lambda e: nlm(e, cfg.nlm_h, cfg.nlm_h_color),
        "bilateral": lambda e: bilateral(e, cfg.bil_d, cfg.bil_sigma_color, cfg.bil_sigma_space),
        "guided": lambda e: guided(e, cfg.gf_r, cfg.gf_eps),
        "bm3d": lambda e: bm3d_filter(e, cfg.bm3d_sigma),
    }

    low_dir = os.path.join(cfg.lol_dir, "low")
    high_dir = os.path.join(cfg.lol_dir, "high")
    rows = []
    files = list_images(low_dir)
    if cfg.limit > 0:
        files = files[:cfg.limit]
    print("device: %s | images: %d | methods: %s | adaptive: %s | ycc_luma: %.2f"
          % (device, len(files), ",".join(methods), cfg.adaptive, cfg.ycc_luma))

    for path in files:
        name = os.path.basename(path)
        hp = os.path.join(high_dir, name)
        if not os.path.exists(hp):
            continue
        x = load_image(path, device)
        gt = to_np(load_image(hp, device))
        with torch.no_grad():
            t0 = time.time()
            _, e_t, _ = net.dce(x)
            t_dce = time.time() - t0
            gain = model_v2.gain_map(x, e_t).squeeze().cpu().numpy()
        base = to_np(e_t)
        masks = RegionMasks(gt, cfg.edge_th, cfg.flat_th)
        # 많이 밝아진 곳일수록 강하게 적용하기 위한 가중치 (--adaptive)
        w = ((gain - gain.min()) / (gain.max() - gain.min() + 1e-8))[..., None].astype(np.float32)

        cache = {}  # 같은 필터를 일반/ycc에서 두 번 계산하지 않도록 (시간은 각각 따로 기록)
        for mth in methods:
            t0 = time.time()
            if mth == "base":
                out = base
            elif mth == "cnn":
                with torch.no_grad():
                    o, _, _ = net.denoise(e_t, x)
                out = to_np(o)
            else:
                fname = mth[4:] if mth.startswith("ycc_") else mth
                if fname not in cache:
                    tf = time.time()
                    cache[fname] = (filters[fname](base), time.time() - tf)
                filtered, f_sec = cache[fname]
                t0 = time.time() - f_sec  # 캐시를 써도 필터 시간은 포함
                out = ycc_combine(base, filtered, cfg.ycc_luma) if mth.startswith("ycc_") else filtered
                if cfg.adaptive:
                    out = (1 - w) * base + w * out
            sec = time.time() - t0  # 노이즈 제거 단계만의 시간 (Zero-DCE 시간 제외)
            r = compute_metrics(out, base, gt, masks)
            r.update(image=name, method=mth, sec=sec)
            rows.append(r)
            if cfg.save_dir:
                d = os.path.join(cfg.save_dir, mth)
                os.makedirs(d, exist_ok=True)
                torchvision.utils.save_image(torch.from_numpy(out).permute(2, 0, 1),
                                             os.path.join(d, os.path.splitext(name)[0] + ".png"))
        print("done: %s (Zero-DCE %.2fs)" % (name, t_dce), flush=True)

    keys = ["psnr", "ssim", "edge_ratio", "flat_luma", "flat_chroma", "sec"]
    print()
    print("%-14s %8s %8s %11s %10s %12s %8s" % ("method", "PSNR", "SSIM", "edge_ratio", "flat_luma", "flat_chroma", "sec"))
    for mth in methods:
        rs = [r for r in rows if r["method"] == mth]
        mean = [np.mean([r[k] for r in rs]) for k in keys]
        print("%-14s %8.3f %8.4f %11.3f %10.2f %12.2f %8.2f" % ((mth,) + tuple(mean)))
    print("\n(n=%d) edge_ratio: 1=경계 유지, 작을수록 흐려짐 | flat_*: 평평한 영역 노이즈(x1000), 낮을수록 좋음"
          % len({r["image"] for r in rows}))

    if cfg.csv:
        os.makedirs(os.path.dirname(cfg.csv) or ".", exist_ok=True)
        with open(cfg.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["image", "method"] + keys)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: (round(r[k], 5) if isinstance(r[k], float) else r[k]) for k in ["image", "method"] + keys})
        print("이미지별 결과 저장:", cfg.csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lol_dir", type=str, default="data/LOL/eval15")
    p.add_argument("--dce_weights", type=str, default="snapshots/Epoch99.pth")
    p.add_argument("--cnn_weights", type=str, default="snapshots/denoise_v1/denoiser_latest.pth")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS),
                   help="쉼표로 구분. 가능: " + ",".join(ALL_METHODS))
    p.add_argument("--ycc_luma", type=float, default=0.0,
                   help="ycc_* 방법에서 밝기에 필터 결과를 섞는 비율 (0=밝기 그대로, 1=일반 필터와 같음)")
    p.add_argument("--adaptive", action="store_true",
                   help="고정 수식 필터를 많이 밝아진 곳일수록 강하게 적용 (gain 맵으로 섞기)")
    p.add_argument("--limit", type=int, default=0, help="앞에서부터 N장만 (0=전체)")
    p.add_argument("--save_dir", type=str, default="", help="방법별 결과 이미지 저장 폴더 (빈 값이면 저장 안 함)")
    p.add_argument("--csv", type=str, default="data/result_compare/metrics.csv")
    # 지표 영역 기준 (정답 이미지를 5x5 평균 후 Sobel 기울기 크기)
    p.add_argument("--edge_th", type=float, default=0.3)
    p.add_argument("--flat_th", type=float, default=0.05)
    # 필터 설정
    p.add_argument("--nlm_h", type=float, default=6, help="NLM 밝기 세기 (0~255 기준)")
    p.add_argument("--nlm_h_color", type=float, default=12, help="NLM 색 세기 (0~255 기준)")
    p.add_argument("--bil_d", type=int, default=9, help="양방향 필터 이웃 지름")
    p.add_argument("--bil_sigma_color", type=float, default=0.1, help="양방향 필터 밝기 차이 허용 (0~1 기준)")
    p.add_argument("--bil_sigma_space", type=float, default=3.0, help="양방향 필터 거리 표준편차 (픽셀)")
    p.add_argument("--gf_r", type=int, default=4, help="가이디드 필터 반경")
    p.add_argument("--gf_eps", type=float, default=0.004, help="가이디드 필터 평활 기준 (분산, 0~1 기준)")
    p.add_argument("--bm3d_sigma", type=float, default=0.0, help="BM3D 노이즈 표준편차 (0~1 기준, 0이면 자동 추정)")
    main(p.parse_args())
