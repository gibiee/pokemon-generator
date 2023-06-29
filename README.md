This is a toy project for implementation of image generating.  
I tested the Pokémon images as a topic.  
I wish this repo could be helpful for many people.

# Preparing dataset

크롤링

# Train / Fine-tuning
I tested 3 different methods.
1. **GAN**  
2. **Transfer learning on Stable Diffusion**  
3. **LORA**


## GAN
GAN can be used to generate image randomly or by class. I trained the dataset on the [StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch.git) model. You can inference by using my checkpoint file.

## Transfer learning on Stable Diffusion
이것은 드림부스 같은 파인튜닝이 아닙니다. Stable Diffusion에 직접적인 학습을 진행했습니다.
(드림부스 논문에서 파인튜닝 방법이라고 했나?)

[ 일단 조사한 부분 메모 ] 
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
로라는 최신 유행하는 방법 중 하나입니다. 빠르고 효과적으로 Stable Diffusion을 파인튜닝할 수 있습니다.



---

### 어느정도 완료되면 0번째 commit으로 초기화 할 것,
