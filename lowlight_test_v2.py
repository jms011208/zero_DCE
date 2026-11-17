"""노이즈 억제망을 붙인 Zero-DCE로 저조도 이미지의 밝기를 개선한다.

예)  python lowlight_test_v2.py --input data/test_data/ --output data/result_v2/ \
         --denoise_weights snapshots/denoise_v1/denoiser_latest.pth

- --input 폴더 아래(하위 폴더 포함)의 모든 이미지를 처리하고, 같은 폴더 구조로 --output 에 PNG로 저장한다.
- --save_extra 를 켜면 비교용으로 기존 Zero-DCE 결과(*_base.png)와 노이즈 마스크(*_mask.png)도 같이 저장한다.
- --no_denoise 를 켜면 기존 Zero-DCE와 같은 결과가 나온다.
"""
import argparse
import os
import time

import torch
import torchvision
from PIL import Image
import numpy as np

import model_v2

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(root):
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower().endswith(IMG_EXT):
                paths.append(os.path.join(dp, fn))
    return sorted(paths)


def load_image(path, device):
    arr = np.asarray(Image.open(path).convert("RGB")) / 255.0
    return torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(device)


def build_net(dce_weights, denoise_weights, device, channels=16):
    net = model_v2.enhance_net_denoise(ch=channels).to(device)
    net.load_dce(dce_weights, map_location=device)
    if denoise_weights:
        net.denoiser.load_state_dict(torch.load(denoise_weights, map_location=device))
    net.eval()
    return net


def run(config):
    device = pick_device()
    use_denoise = not config.no_denoise
    if use_denoise and not config.denoise_weights:
        raise SystemExit("--denoise_weights 를 지정하거나 --no_denoise 를 사용하세요.")
    net = build_net(config.dce_weights, config.denoise_weights if use_denoise else None,
                    device, config.channels)

    files = list_images(config.input)
    print("device: %s | images: %d" % (device, len(files)))
    with torch.no_grad():
        for path in files:
            x = load_image(path, device)
            start = time.time()
            e, out, m, _ = net(x)
            if not use_denoise:
                out = e
            elapsed = time.time() - start

            rel = os.path.relpath(path, config.input)
            stem = os.path.splitext(rel)[0]
            dst = os.path.join(config.output, stem + ".png")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            torchvision.utils.save_image(out, dst)
            if config.save_extra:
                torchvision.utils.save_image(e, os.path.join(config.output, stem + "_base.png"))
                torchvision.utils.save_image(m, os.path.join(config.output, stem + "_mask.png"))
            print("%s  (%.3fs)" % (rel, elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/test_data/")
    parser.add_argument("--output", type=str, default="data/result_v2/")
    parser.add_argument("--dce_weights", type=str, default="snapshots/Epoch99.pth")
    parser.add_argument("--denoise_weights", type=str, default="snapshots/denoise_v1/denoiser_latest.pth")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--no_denoise", action="store_true")
    parser.add_argument("--save_extra", action="store_true")
    run(parser.parse_args())
