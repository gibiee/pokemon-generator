# 현재 개발 중 입니다.
- stylegan3-fun/projector.py 코드를 리펙토링하기 -> dataset 이미지에 대해서 class 폴더로 분류하기
- 훈련 중간결과 정리해서 비교하고 best weight 선별

## Process
1. Environment setup
2. Preapare dataset
3. Model training
4. Inference (Web Demo)

## Environment setup
```sh
conda create -n test python=3.9
conda activate test

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install ipykernel

pip install click tqdm scipy
pip install imageio[ffmpeg] imageio[pyav]

pip install selenium undetected-chromedriver
```

### Code
```sh
git clone https://github.com/gibiee/pokemon-generator.git
cd pokemon-generator

git clone https://github.com/NVlabs/stylegan3.git
git clone https://github.com/PDillis/stylegan3-fun
```


## Preapare dataset 

### Crawling pokemon images : `1_crawling.py`
- 총 1025장의 이미지를 크롤링 : https://www.pokemon.com/us/pokedex
- 수집한 이미지를 512x512 해상도로 저장 (StyleGan2 모델의 사이즈 규칙 때문에)
- zip 파일으로 만드는 과정 필요

### Make json file for model training : `2_make_json.py`
- `crawling.py`에서 저장한 `info.csv` 파일을 참고하여 `.json` 파일 제작
- 그 후 모델 학습에 적합한 `.zip` 파일으로 만드는 과정 필요
```sh
python stylegan3/dataset_tool.py \
    --source ./dataset \
    --dest ./dataset/dataset.zip
```

## Model training
I trained the dataset on the [StyleGAN3](https://github.com/NVlabs/stylegan3.git) model.

- 00000 : conditional training
```sh
CUDA_VISIBLE_DEVICES=0 python stylegan3/train.py \
    --gpus=1 \
    --cfg=stylegan3-r \
    --data=./dataset/dataset.zip \
    --outdir=./stylegan3/training-runs \
    --batch=4 \
    --mirror=1 \
    --cond=1 \
    --gamma=8 \
    --kimg=25000 \
    --tick=4 \
    --snap=10

# --kimg KIMG : Total training duration  [default: 25000]
# --tick KIMG : How often to print progress  [default: 4]
# --snap TICKS : How often to save snapshots  [default: 50]
```

- 00001 : 위 버전이 tick/snap 간격이 너무 짧아서 resume한 버전
```sh
CUDA_VISIBLE_DEVICES=0 python stylegan3/train.py \
    --resume=./stylegan3/training-runs/00000-stylegan3-r-dataset-gpus1-batch4-gamma8/network-snapshot-001200.pkl \
    --gpus=1 \
    --cfg=stylegan3-r \
    --data=./dataset/dataset.zip \
    --outdir=./stylegan3/training-runs \
    --batch=4 \
    --mirror=1 \
    --cond=1 \
    --gamma=8 \
    --kimg=25000 \
    --tick=10 \
    --snap=10
```

- 00002 : conditional 학습이 잘 안되어서 non-conditional 학습
```sh
CUDA_VISIBLE_DEVICES=0 python stylegan3/train.py \
    --gpus=1 \
    --cfg=stylegan3-r \
    --data=./dataset/dataset.zip \
    --outdir=./stylegan3/training-runs \
    --batch=4 \
    --mirror=1 \
    --gamma=8 \
    --kimg=25000 \
    --tick=10 \
    --snap=10 \
    --aug=ada \
    --dry-run
```

- 00003 : augment_kwargs 수정 후 non-conditional 학습
  - https://medium.com/@Dok11/how-to-check-augmentations-for-the-stylegan3-196f8c2ddf07 참고
  - `train.py` 256 line : `c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=0, xint=1, xint_max=0.1, scale=0.5, rotate=0.3, aniso=1, xfrac=0.2, brightness=0.3, contrast=0.2, lumaflip=0, hue=0, saturation=0.2)`

- 00004 : 위 00003 버전의 augment_kwargs 적용하면서 conditional 학습
```sh
CUDA_VISIBLE_DEVICES=0 python stylegan3/train.py \
    --gpus=1 \
    --cfg=stylegan3-r \
    --data=./dataset/dataset.zip \
    --cond=1 \
    --outdir=./stylegan3/training-runs \
    --batch=4 \
    --mirror=1 \
    --gamma=8 \
    --kimg=25000 \
    --tick=10 \
    --snap=10 \
    --aug=ada \
    --dry-run
```

- 00005 : 위 00004 버전의 중간결과가 좋지 않으므로, 00003 버전의 학습을 resume
```sh
CUDA_VISIBLE_DEVICES=0 python stylegan3/train.py \
    --resume=./stylegan3/training-runs/00003-stylegan3-r-dataset-gpus1-batch4-gamma8/network-snapshot-003800.pkl \
    --gpus=1 \
    --cfg=stylegan3-r \
    --data=./dataset/dataset.zip \
    --outdir=./stylegan3/training-runs \
    --batch=4 \
    --mirror=1 \
    --gamma=8 \
    --kimg=25000 \
    --tick=10 \
    --snap=10 \
    --aug=ada \
    --dry-run
```


## Projection
```sh
CUDA_VISIBLE_DEVICES=1 python 3_projection_fun.py

CUDA_VISIBLE_DEVICES=1 python 3_projection.py
```

## Inference : `demo.py`
You can generate a image randomly or by class in [web demo](#web-demo).
