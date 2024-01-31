# Currently in development 

# Web Demo

# Preparing dataset
`1_crawling.py`  
https://pokemonkorea.co.kr/pokedex 이 사이트를 2023-12-22 에 크롤링  
- 사이트 전체 이미지 1229장  
- 크롤링 결과 이미지 1195장  

- "거다이맥스" 폼은 일반적인 포켓몬 이미지와 특성이 다르다고 판단하여 제외 (총 34장)  
- 흰색 배경의 깔끔한 RGB 이미지를 얻기 위하여 추가적인 처리  

|Before|After|
|:---:|:---:|
|![before](https://github.com/gibiee/pokemon-generator/assets/37574274/04b9914e-56e6-43b2-a5cb-61aff138fadd)|![after](https://github.com/gibiee/pokemon-generator/assets/37574274/bfb17068-9ae3-4b08-8b69-731a1a6efd7f)|


# Train
I trained the dataset on the [StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch.git) model in Colab.  


# Inference
You can generate a image randomly or by class in [web demo](#web-demo).

