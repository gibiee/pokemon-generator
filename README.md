# Currently in development 
## To do list
- [ ] https://zukan.pokemon.co.jp 해당 사이트에서 크롤링하도록 코드 수정
- [ ] colab이 아닌, 로컬 머신에서 train 코드 돌리기

# Web Demo

# Process
1. Preparing dataset
2. Train
3. Inference

## Preparing dataset : `crawling.py`
- https://pokemonkorea.co.kr/pokedex 이 사이트를 크롤링
- 수집한 이미지를 512x512 해상도로 저장 (모델의 사이즈 규칙 때문에)
- 크롤링 코드 상에서, "거다이맥스" 폼은 일반적인 포켓몬 이미지와 특성이 다르다고 판단하여 제외 (총 34장)
- 흰색 배경의 깔끔한 RGB 이미지를 얻기 위하여 추가적인 처리

|Before|After|
|:---:|:---:|
|![before](https://github.com/gibiee/pokemon-generator/assets/37574274/04b9914e-56e6-43b2-a5cb-61aff138fadd)|![after](https://github.com/gibiee/pokemon-generator/assets/37574274/bfb17068-9ae3-4b08-8b69-731a1a6efd7f)|

- 사이트 전체 이미지 1252장
- 크롤링 결과 이미지 1195장


## Train
I trained the dataset on the [StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch.git) model.

### Installation
```sh
```

### Run train
```sh
```


## Inference : `demo.py`
You can generate a image randomly or by class in [web demo](#web-demo).

