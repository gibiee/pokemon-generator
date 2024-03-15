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

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install ipykernel
pip install selenium undetected-chromedriver pandas
pip install requests click psutil
```

- ImportError: No module named 'upfirdn2d_plugin' 문제 해결!

## Preparing dataset : `crawling.py`
- 총 1025장의 이미지를 크롤링 : https://www.pokemon.com/us/pokedex
- 수집한 이미지를 512x512 해상도로 저장 (StyleGan2 모델의 사이즈 규칙 때문에)
- zip 파일으로 만드는 과정 필요


## Train
I trained the dataset on the [StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch.git) model.

### 

### Run train
1. (Line 271 in `train.py`) `augpipe_specs = {}` 코드에 아래 설정을 추가  
`'custom': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),`

`'custom': dict(xflip=1, rotate90=0.2, xint=0.5, scale=0.2, rotate=1, aniso=1, xfrac=0.8, brightness=0.8, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),`

`'custom': dict(xflip=1, xint=1, scale=1, rotate=0.2, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),`

2. Run command
```sh
CUDA_VISIBLE_DEVICES=1 python stylegan2-ada-pytorch/train.py \
    --data=dataset/images \
    --mirror=true \
    --augpipe=custom \
    --outdir=./training-runs \
    --dry-run
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

