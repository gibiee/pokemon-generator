# 현재 개발 진행 중 입니다...
- 속성 슬라이더 -1 ~ 1 으로 수정
- w 랜덤 기능은 그 이미지가 균일하게 나오므로, 삭제하는게 더 나을수도


# Index
- Demo
- Development
  - Environment setup
  - Preparing dataset
  - Inversion
  - Image Generation
  
  Projections by StyleGAN-XL


# Demo

## HuggingFace 데모 준비 중


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
- 웹 사이트에서 총 1025장의 이미지를 크롤링 : https://www.pokemon.com/us/pokedex
- 수집한 이미지를 1024x1024 해상도로 resize 후 저장
- 속성(type) 정보를 표로 정리하여 `info.csv` 파일으로 저장


## Inversion

### Run inversion of the dataset

```sh
python run_inversion_custom.py
```

## Image Generation
