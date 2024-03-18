# Currently in development 

환경 세팅이 잘 안되므로 도커로 해야할 것 같다...
StyleGAN-XL -> 도커가 없다.
StyleGAN3 -> 이걸로 하자!

nvidia-docker부터 설치 : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

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
```

### Run Docker
```sh
docker build --tag stylegan3 .

```

## Installation
```sh
conda create -n pokemon python=3.8
conda activate pokemon

# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install ipykernel
pip install selenium undetected-chromedriver requests pandas
```

## Preparing dataset : `crawling.py`
- 총 1025장의 이미지를 크롤링 : https://www.pokemon.com/us/pokedex
- 수집한 이미지를 512x512 해상도로 저장 (StyleGan2 모델의 사이즈 규칙 때문에)
- zip 파일으로 만드는 과정 필요

## Train
I trained the dataset on the [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl) model.

```sh
CUDA_VISIBLE_DEVICES=1 python train.py \
    --outdir=../training-runs/pokemon \
    --cfg=stylegan3-t \
    --data=../dataset/images \
    --mirror=true \
    --gpus=1 \
    --batch=16 \
    --snap 10 \
    --kimg 100000 \
    --syn_layers 10 \
    --dry-run
```
```sh
CUDA_VISIBLE_DEVICES=1 python stylegan2-ada-pytorch/train.py \
    --data=dataset/images \
    --mirror=true \
    --augpipe=custom \
    --outdir=./training-runs \
    --gpus=1 \
    
```

```sh
python stylegan2-ada-pytorch/train.py \
    --data=dataset/images \
    --mirror=true \
    --augpipe=custom \
    --outdir=./training-runs \
    --snap=100 \
    --batch 4 \
    --dry-run
```


## Inference : `demo.py`
You can generate a image randomly or by class in [web demo](#web-demo).

