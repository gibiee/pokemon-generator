import torch
import numpy as np
import pandas as pd
from PIL import Image
import random

import sys
sys.path.append('./stylegan-xl')
from torch_utils import gen_utils
import dnnlib, legacy

NETWORK = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl"
SEED = random.randint(0, 1000000)
TRUNCATION_PSI = 1.0

np.random.seed(SEED)
torch.manual_seed(SEED)

# Load networks.
print('Loading networks from "%s"...' % NETWORK)
device = torch.device('cuda')
with dnnlib.util.open_url(NETWORK) as f:
    G = legacy.load_network_pkl(f)['G_ema']
    G = G.eval().requires_grad_(False).to(device)

# -------------------------------------------------------------------------
# 특정 class에 해당하는 샘플을 선별

class_en = ["Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]

class_name = class_en[1]
print(f'class_name : {class_name}')

info_csv = pd.read_csv('dataset/info.csv', index_col=0)
target_samples = info_csv[info_csv[class_name] == 1]
sample_index = random.choice(target_samples.index)
sample_key = sample_index + 1
sample_fn = str(sample_key).zfill(4)
Image.open(f'dataset/images_1024/{sample_fn}.jpg')

sample_vector = np.load(f'projections/default/{sample_fn}_projected_w.npz')['w']
sample_w = torch.from_numpy(sample_vector).to(device)
sample_w = sample_w.repeat(1, G.num_ws, 1)
sample_output = gen_utils.w_to_img(G, sample_w, to_np=True)[0]
Image.fromarray(sample_output, 'RGB')

class_vector = np.load(f'class_vectors/{class_name}.npy')
class_w = torch.from_numpy(class_vector).to(device)
class_w = class_w.repeat(1, G.num_ws, 1)
class_output = gen_utils.w_to_img(G, class_w, to_np=True)[0]
Image.fromarray(class_output, 'RGB')

# -------------------------------------------------------------------------
# (랜덤, 샘플, 클래스) 3가지 벡터에 가중치를 적용하여 생성

random_seed = random.randint(0, 2**32 - 1)
random_w = gen_utils.get_w_from_seed(G, batch_sz=1, device=device, seed=random_seed)
random_output = gen_utils.w_to_img(G, random_w, to_np=True)[0]
Image.fromarray(random_output, 'RGB')


weight = 0.4
w = sample_w * weight + random_w * (1-weight)
# w = sample_w * 0.4 + random_w * 0.4 + class_w * 0.2
output = gen_utils.w_to_img(G, w, to_np=True)[0]
Image.fromarray(output, 'RGB')

# 특정 샘플의 w와 랜덤 샘플의 w를 섞어서 생성했을 때, 그 두 샘플의 특징이 조화롭게 반영되지 않음.
# 벡터 공간에서의 w가 특징을 잘 반영하지 못하는 듯함.