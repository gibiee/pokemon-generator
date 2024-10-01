<div align="center">
<h1>Pokémon Generator</h1>
<a href='https://huggingface.co/spaces/InstantX/InstantID'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
</div>



<details>
<summary>More Information</summary>

# Development

## Installation

### Environment Setup
```sh
conda env create -f stylegan-xl/environment.yml

conda install pandas pillow tdqm
pip install selenium undetected-chromedriver
```

## Preparing Dataset

### Crawling Pokémon images : `crawling.py`
- GUI 환경에서 실행하기를 권장
- 웹 사이트에서 총 1025장의 이미지를 크롤링 : https://www.pokemon.com/us/pokedex
- 수집한 이미지를 1024x1024 해상도로 resize 후 저장
- 속성(type) 정보를 표로 정리하여 `info.csv` 파일으로 저장

## Feature Extraction

### Run inversion of the dataset : `run_inversion_custom.py`
- StyleGAN-XL 기반의 inversion 진행
- PTI 옵션을 적용하지 않을 때, 범용성이 더 높음.

## Inference

### Generate a Pokémon image
- base 포켓몬 이미지를 먼저 생성한 후, 이를 기반으로 class 정보(feature)를 조금씩 수정하면서 편집

# To-Do List
- huggingspace 코드 상에서 zeroGPU 적용
- 데모 스크린샷 첨부
- 데모 사용법 작성
- 예시 이미지 선별 후 첨부
</details>