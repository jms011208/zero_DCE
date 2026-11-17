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
