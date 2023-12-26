This is a personal toy project for implementation of image generation.  
I choose the Pokémon as a topic.  
I wish this repo could be helpful for many people.  

# Preparing dataset
https://pokemonkorea.co.kr/pokedex 이 사이트를 2023-12-22 에 크롤링  
사이트 총 이미지 1229장  
크롤링 총 이미지 1195장  

- "거다이맥스" 폼은 일반적인 포켓몬 이미지와 특성이 다르다고 판단하여 제외 (총 34장)  
- 흰색 배경의 RGB 이미지를 얻기 위하여 추가적인 처리  

|Before|After|
|:---:|:---:|
|![before](https://github.com/gibiee/pokemon-generator/assets/37574274/04b9914e-56e6-43b2-a5cb-61aff138fadd)|![after](https://github.com/gibiee/pokemon-generator/assets/37574274/bfb17068-9ae3-4b08-8b69-731a1a6efd7f)|


# Train / Fine-tuning
I tested 3 different methods.
1. **GAN**  
2. **Transfer learning on Stable Diffusion**  
3. **LORA**

## GAN
GAN can be used to generate image randomly or by class. I trained the dataset on the [StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch.git) model. You can inference by using my checkpoint file.

## Transfer learning on Stable Diffusion

[ 조사내용 메모 ] 
1. release 전용 Stable Diffusion repo  
https://github.com/CompVis/stable-diffusion  
train -t 옵션이 있기는 한데, 이걸로 train 할 수 있을지는 모르겠음. 그리고 config 폴더에 train 관련 파일이 없음.

2. 개발 전용 Stable Diffusion repo  
https://github.com/pesser/stable-diffusion  
config 파일들이 많이 준비되어 있음

3. 포켓몬 파인튜닝에 대한 자료  
https://github.com/justinpinkney/stable-diffusion  
특히 pokemon.yaml 파일을 직접 작성하였음. 그런데 huggingface의 "lambdalabs/pokemon-blip-captions"를 거쳐서 학습데이터를 로드하기 때문에 수정 필요.


## LORA
로라는 최신 유행하는 방법 중 하나입니다. 빠르고 효과적으로 Stable Diffusion을 파인튜닝할 수 있다.

https://github.com/cloneofsimo/lora.git

