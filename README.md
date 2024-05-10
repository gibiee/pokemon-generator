# 현재 개발 중 입니다.

## Process
1. Environment setup
2. Preapare dataset
3. Model training
4. Inference (Web Demo)

## Environment setup
```sh
git clone https://github.com/gibiee/pokemon-generator.git
git clone https://github.com/NVlabs/stylegan3.git

cd stylegan3

conda env create -f environment.yml -n pokemon
conda activate pokemon

conda install ipykernel
pip install numpy==1.22.4
pip install selenium undetected-chromedriver requests pandas psutil
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

- Conditional 학습이 잘 안되어서 전체 데이터로 학습
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
    --snap=10
```

## Inference : `demo.py`
You can generate a image randomly or by class in [web demo](#web-demo).
