# 사전학습 생성 테스트
CUDA_VISIBLE_DEVICES=1 python gen_images.py --outdir=out --trunc=0.7 --seeds=10-15 --batch-sz 1 --network=https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon256.pkl

# 학습 테스트
CUDA_VISIBLE_DEVICES=0 python train.py \
  --gpus=1 \
  --cfg=stylegan3-t \
  --data=../dataset/dataset.zip \
  --cond=1 \
  --outdir=./training-runs/pokemon \
  --batch=4 \
  --mirror=1 \

  --cbase 16384 \
  --cmax 256 \
  --syn_layers 7 \

  --kimg=20000 \
  --tick=10 \
  --snap=10 \

  --up_factor=4 \
  --dry-run





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