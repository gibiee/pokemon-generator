import torch
import numpy as np
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
# 특정 class의 이미지 생성

class_en = ["Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]

class_name = class_en[1]
print(f'class_name : {class_name}')

vector_path = f'class_vectors/{class_name}.npy'
projected_w = np.load(vector_path)
projected_w = torch.from_numpy(projected_w).to(device)
projected_w.shape # (1, 1, 512)

# projected_w = projected_w[0]
# projected_w.repeat(1, G.num_ws, 1).shape

projected_w = projected_w.repeat(1, G.num_ws, 1)
output = gen_utils.w_to_img(G, projected_w, to_np=True)[0]
Image.fromarray(output, 'RGB')

# -------------------------------------------------------------------------
# 각 class에 가중치를 적용하여 생성

normal = round(random.random(), 2)
fire = round(random.random(), 2)
water = round(random.random(), 2)
grass = round(random.random(), 2)
electric = round(random.random(), 2)
ice = round(random.random(), 2)
fighting = round(random.random(), 2)
poison = round(random.random(), 2)
ground = round(random.random(), 2)
flying = round(random.random(), 2)
psychic = round(random.random(), 2)
bug = round(random.random(), 2)
rock = round(random.random(), 2)
ghost = round(random.random(), 2)
dragon = round(random.random(), 2)
dark = round(random.random(), 2)
steel = round(random.random(), 2)
fairy = round(random.random(), 2)

values = [normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark,steel, fairy]
values = np.array(values)
values_normalized = values / values.sum()

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
