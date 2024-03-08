# Currently in development 

## Process
1. Environment
2. Crwaling images
3. Train
4. Inference (Web Demo)

## Environment

### Code clone
```sh
git clone https://github.com/gibiee/pokemon-generator.git
cd pokemon-generator
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

## Installation
```sh
conda create -n pokemon python=3.8
conda activate pokemon

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install ipykernel
pip install selenium pandas
```

## Preparing dataset : `crawling.py`
- https://www.pokemon.com/us/pokedex 이 사이트를 크롤링
- 수집한 이미지를 512x512 해상도로 저장 (StyleGan2 모델의 사이즈 규칙 때문에)
- ~~크롤링 코드 상에서, "거다이맥스" 폼은 일반적인 포켓몬 이미지와 특성이 다르다고 판단하여 제외 (총 34장)~~
- 흰색 배경의 깔끔한 RGB 이미지를 얻기 위하여 추가적인 처리

|Before|After|
|:---:|:---:|
|![before](https://github.com/gibiee/pokemon-generator/assets/37574274/04b9914e-56e6-43b2-a5cb-61aff138fadd)|![after](https://github.com/gibiee/pokemon-generator/assets/37574274/bfb17068-9ae3-4b08-8b69-731a1a6efd7f)|

- 사이트 전체 이미지 1252장
- 크롤링 결과 이미지 1195장


## Train
I trained the dataset on the [StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch.git) model.

### 

### Run train
1. (Line 271 in `train.py`) `augpipe_specs = {}` 코드에 아래 설정을 추가  
`'custom': dict(xflip=1, rotate90=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1)`

2. Run command
```sh
python stylegan2-ada-pytorch/train.py \
    --data=dataset/images \
    --mirror=true \
    --augpipe=blit \
    --outdir=./training-runs \
    --snap=100 \
    --gpus=1 \
    --batch 4 \
    --dry-run
```


## Inference : `demo.py`
You can generate a image randomly or by class in [web demo](#web-demo).

