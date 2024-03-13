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
pip install selenium undetected-chromedriver pandas
```

## Preparing dataset : `crawling.py`
- https://www.pokemon.com/us/pokedex 이 사이트를 크롤링
- 수집한 이미지를 512x512 해상도로 저장 (StyleGan2 모델의 사이즈 규칙 때문에)
- 이미지 총 1026장


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

