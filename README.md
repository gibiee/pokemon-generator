# 현재 개발 진행 중 입니다...
- inversion : G vs 갱신된 G 결과 비교하기

# Index
- Demo
- Development
  - Environment setup
  - Preparing dataset
  - Projections by StyleGAN-XL


# Demo

# Development

## Environment setup
```sh
conda env create -f stylegan-xl/environment.yml

conda install pandas pillow tdqm
pip install selenium undetected-chromedriver
# brew install chromedriver
```

## Preparing dataset

### Crawling pokemon images : `crawling_images.py`
- GUI 환경에서 실행하기를 권장
- 총 1025장의 이미지를 크롤링 : https://www.pokemon.com/us/pokedex
- 수집한 이미지를 1024x1024 해상도로 resize 후 저장
- 속성(type) 정보를 표로 정리하여 `.csv` 파일으로 저장


## Projections by StyleGAN-XL

```sh
CUDA_VISIBLE_DEVICES=1 python run_inversion_custom.py
```