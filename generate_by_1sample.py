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

class_name = class_en[9]
print(f'class_name : {class_name}')

info_csv = pd.read_csv('dataset/info.csv', index_col=0)
info_csv.head()
info_csv.tail()

target_samples = info_csv[info_csv[class_name] == 1]
sample_index = random.choice(target_samples.index)
sample_key = sample_index + 1
sample_fn = str(sample_key).zfill(4)
Image.open(f'dataset/images_512/{sample_fn}.jpg')

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









target_vector = None
for i, value_normalized in enumerate(values_normalized) :
    class_name = class_en[i]
    class_feature = np.load(f'class_features/{class_name}.npy')

    if target_vector is None :
        target_vector = class_feature * value_normalized
    else :
        target_vector += class_feature * value_normalized

projected_w = target_vector.copy()
projected_w = torch.from_numpy(projected_w).to(device)
projected_w.shape # (1, 1, 512)

projected_w = projected_w.repeat(1, G.num_ws, 1)
output = gen_utils.w_to_img(G, projected_w, to_np=True)[0]
Image.fromarray(output, 'RGB')

## class vector를 그대로 사용하면 예상과 다르게 다양성이 없어지는 문제 발생
## 따라서, 베이스 벡터를 사용하여 베이스 이미지를 생성한 후 작업하는 방향으로 변경

# -------------------------------------------------------------------------
# 무작위로 베이스 벡터를 통한 이미지 생성
SEED = random.randint(0, 1000000)

base_w = gen_utils.get_w_from_seed(G, batch_sz=1, device=device, seed=SEED)
base_w.shape

output = gen_utils.w_to_img(G, base_w, to_np=True)[0]
output.shape

output_img = Image.fromarray(output, 'RGB')
output_img

# -------------------------------------------------------------------------
# 위 베이스 벡터를 기반으로 클래스 벡터를 적용

base_ratio = 0.5
edit_w = (base_w * base_ratio) + projected_w * (1 - base_ratio)

output_edit = gen_utils.w_to_img(G, edit_w, to_np=True)[0]
output_edit.shape

Image.fromarray(output_edit, 'RGB')
