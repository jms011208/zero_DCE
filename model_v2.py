"""Zero-DCE + 노이즈 억제 소형 신경망 (NoiseSuppressor).

기존 DCE-Net(`model.enhance_net_nopool`)은 그대로 재사용하고, 그 출력(밝아진 이미지) 뒤에
작은 CNN을 붙인다. 이 CNN은 픽셀마다
  - M (1채널, 0~1): "이 픽셀에 증폭된 노이즈가 많다"는 확신도 (노이즈 마스크)
  - R (3채널)     : 그 픽셀에서 빼야 할 노이즈 성분 (잔차)
를 예측하고, 최종 출력은  out = E - M * R  이다.  (E = DCE-Net이 만든 밝아진 이미지)
M이 0이면 그 픽셀은 원래 Zero-DCE 결과 그대로 유지되고, M이 클수록 많이 보정된다.

입력으로는 E(3채널) 외에 "이 픽셀이 얼마나 많이 밝아졌는가"를 나타내는 gain 맵(1채널)을 같이 준다.
어두운 곳일수록 많이 증폭되고 노이즈도 같이 증폭되기 때문에, 노이즈가 심한 위치를 찾는 단서가 된다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import enhance_net_nopool


def gain_map(x, e, k=7):
    """밝기 증폭 정도 맵 (N,1,H,W), 값 범위 0~1.

    x: 입력 저조도 이미지, e: DCE-Net이 밝게 만든 이미지. 둘 다 (N,3,H,W), 0~1.
    평균 풀링으로 부드럽게 만든 뒤 비율(e/x)을 구하고 log 스케일로 0~1로 눌러준다.
    """
    pad = k // 2
    ex = F.avg_pool2d(e.mean(1, keepdim=True), k, 1, pad, count_include_pad=False)
    xx = F.avg_pool2d(x.mean(1, keepdim=True), k, 1, pad, count_include_pad=False)
    g = ex / (xx + 1e-3)
    return torch.log1p(g.clamp(max=50.0)) / math.log(51.0)


def _conv(in_ch, out_ch, dilation=1):
    return nn.Conv2d(in_ch, out_ch, 3, 1, padding=dilation, dilation=dilation, bias=True)


class NoiseSuppressor(nn.Module):
    """노이즈 마스크 M과 잔차 R을 예측하는 작은 CNN (기본 약 6천 개 파라미터)."""

    def __init__(self, ch=16, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.relu = nn.ReLU(inplace=True)
        self.c1 = _conv(4, ch)                 # 입력: E(3) + gain(1)
        self.c2 = _conv(ch, ch, dilation=2)    # 넓은 범위(9x9)를 보기 위한 dilation
        self.c3 = _conv(ch, ch)
        self.mask_head = _conv(ch, 1)
        self.res_head = _conv(ch, 3)
        # 처음에는 "아무것도 안 고치는" 상태에서 시작 (M ~ 0.12, R ~ 0)
        nn.init.constant_(self.mask_head.bias, -2.0)
        nn.init.normal_(self.res_head.weight, 0.0, 1e-3)
        nn.init.zeros_(self.res_head.bias)

    def forward(self, e, g):
        f1 = self.relu(self.c1(torch.cat([e, g], 1)))
        f2 = self.relu(self.c2(f1))
        f3 = self.relu(self.c3(f2)) + f1
        m = torch.sigmoid(self.mask_head(f3))
        r = torch.tanh(self.res_head(f3)) * self.res_scale
        out = (e - m * r).clamp(0.0, 1.0)
        return out, m, r


class enhance_net_denoise(nn.Module):
    """DCE-Net (기존) + NoiseSuppressor (신규)."""

    def __init__(self, ch=16, res_scale=0.2):
        super().__init__()
        self.dce = enhance_net_nopool()
        self.denoiser = NoiseSuppressor(ch=ch, res_scale=res_scale)

    def load_dce(self, path, map_location="cpu"):
        self.dce.load_state_dict(torch.load(path, map_location=map_location))

    def freeze_dce(self):
        for p in self.dce.parameters():
            p.requires_grad = False
        self.dce.eval()

    def denoise(self, e, x):
        """이미 밝아진 이미지 e(와 원본 x)에서 노이즈 억제만 수행."""
        return self.denoiser(e, gain_map(x, e))

    def forward(self, x):
        """반환: (e, out, m, r)
        e   : 기존 Zero-DCE 결과 (노이즈 억제 전)
        out : 노이즈 억제까지 마친 최종 결과
        m   : 노이즈 마스크, r : 곡선 파라미터 맵 (기존 Zero-DCE의 r)
        """
        _, e, r = self.dce(x)
        out, m, _ = self.denoise(e, x)
        return e, out, m, r
