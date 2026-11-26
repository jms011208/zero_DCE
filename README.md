# zero_DCE

Zero-DCE(Zero-Reference Deep Curve Estimation)로 저조도 이미지의 밝기를 복원하고, 모델을 개선하는 프로젝트입니다.

- 원본 코드: https://github.com/Li-Chongyi/Zero-DCE (CVPR 2020) — 비상업적 연구 목적(CC BY-NC 4.0). 원본 README는 `ORIGINAL_README.md` 참고
- 사전학습 가중치: `snapshots/Epoch99.pth`
- 데이터셋(`data/`)은 용량 때문에 Git에 올리지 않습니다. 받는 방법은 `zero_DCEob/04 데이터셋과 모델.md` 참고
- 프로젝트 노트(규칙, 진행사항, 수정사항)는 Obsidian 보관함 `zero_DCEob/`에 있습니다

## Colab에서 시작하기
```python
!git clone https://github.com/jms011208/zero_DCE.git
%cd zero_DCE
```
이후 데이터 다운로드 코드는 `zero_DCEob/04 데이터셋과 모델.md`의 "Colab에서 데이터 받기"를 실행하세요.

## 노이즈 억제망 v2 (브랜치 `improve/denoise-net`)
밝기 개선 때 증폭되는 노이즈를, DCE-Net 뒤에 붙인 소형 신경망(약 5.8k 파라미터)이 보정합니다. 설계와 결과는 `zero_DCEob/03 수정사항.md` 참고.
```bash
# 1) 노이즈 억제망 학습 (DCE-Net은 Epoch99.pth로 고정)
python lowlight_train_v2.py --snapshots_folder snapshots/denoise_v1/ --num_epochs 30
# 2) LOL eval15로 기존 Zero-DCE와 비교 (PSNR/SSIM)
python eval_lol.py --lol_dir data/LOL/eval15 --denoise_weights snapshots/denoise_v1/denoiser_latest.pth
# 3) 이미지 폴더에 적용 (--save_extra: 기존 결과와 노이즈 마스크도 저장)
python lowlight_test_v2.py --input data/test_data/ --output data/result_v2/ --save_extra
```

## 노이즈 제거 방법 비교 (`compare_denoise.py`)
같은 Zero-DCE 결과에 노이즈 억제 CNN과 고정 수식 필터(비지역 평균, 양방향, 가이디드, BM3D)를 각각 적용해 LOL eval15에서 같은 지표로 비교합니다.
```bash
pip install bm3d          # BM3D를 쓸 때만 (없으면 자동으로 건너뜀)
python compare_denoise.py --cnn_weights snapshots/denoise_v1/denoiser_latest.pth --save_dir data/result_compare
```
- 지표: PSNR, SSIM, 경계 유지율(`edge_ratio`, 1=유지), 평평한 영역 밝기/색 노이즈(`flat_luma`, `flat_chroma`, 낮을수록 좋음), 처리 시간
- `--adaptive`: 고정 필터를 많이 밝아진 곳일수록 강하게 적용
- `ycc_nlm`, `ycc_bilateral`, `ycc_guided`, `ycc_bm3d`: 밝기는 Zero-DCE 결과 그대로 두고 색(Cb, Cr) 노이즈만 필터로 제거 (`--ycc_luma`로 밝기에 필터를 섞는 비율 조절)
- 이미지별 결과는 `data/result_compare/metrics.csv`에 저장
