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

### Crawling pokemon images : `crawling.py`
- 총 1025장의 이미지를 크롤링 : https://www.pokemon.com/us/pokedex
- 수집한 이미지를 512x512 해상도로 저장 (StyleGan2 모델의 사이즈 규칙 때문에)
- zip 파일으로 만드는 과정 필요

### Make json file for model training : `make_json.py`
- `crawling.py`에서 저장한 `info.csv` 파일을 참고하여 `.json` 파일 제작
- 그 후 모델 학습에 적합한 `.zip` 파일으로 만드는 과정 필요

```sh
# In "stylegan3" folder
python dataset_tool.py \
    --source ../dataset/images \
    --dest ../dataset/dataset.zip
```

## Model training
I trained the dataset on the [StyleGAN3](https://github.com/NVlabs/stylegan3.git) model.

```sh
CUDA_VISIBLE_DEVICES=0 python train.py \
    --gpus=1 \
    --cfg=stylegan3-t \
    --data=../dataset/images \
    --outdir=./training-runs \
    --batch=4 \
    --gamma=8.2 \
    --mirror=1
```

## Inference : `demo.py`
You can generate a image randomly or by class in [web demo](#web-demo).
