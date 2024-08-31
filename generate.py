import sys
sys.path.append('./stylegan-xl')

import os, glob, copy
import dill, imageio
import numpy as np
from PIL import Image
import torch
import dnnlib, legacy

NETWORK = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# Load networks.
print('Loading networks from "%s"...' % NETWORK)
device = torch.device('cuda')
with dnnlib.util.open_url(NETWORK) as f:
    G = legacy.load_network_pkl(f)['G_ema']
    G = G.eval().requires_grad_(False).to(device)

# -------------------------------------------------------------------------

vector_path = 'class_vector/Fairy.npy'
projected_w = np.load(vector_path)
projected_w = torch.from_numpy(projected_w).to(device)
projected_w.shape # (1, 1, 512)

# projected_w = projected_w[0]
# projected_w.repeat(1, G.num_ws, 1).shape

synth_image = G.synthesis(projected_w.repeat(1, G.num_ws, 1))
synth_image = (synth_image + 1) * (255/2)
synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
Image.fromarray(synth_image, 'RGB')

# -------------------------------------------------------------------------

class_en = ["Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]

normal = 0
fire = 0.1
water = 0.12
grass = 1
electric = 0.1
ice = 0.1
fighting = 0.1
poison = 0
ground = 0.1
flying = 0
psychic = 0
bug = 0.1
rock = 0
ghost = 0
dragon = 0.1
dark = 1
steel = 0.1
fairy = 0.1

values = [normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark,steel, fairy]
values = np.array(values)
values_normalized = values / values.sum()

# -------------------------------------------------------------------------
# class vector를 그대로 사용하면 생각보다 생성 퀄리티가 별로라 베이스가 되는 벡터가 있는게 낫지 않나..?

base_vector = 
w_avg_samples = 10000
print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
z_samples = torch.from_numpy(np.random.RandomState(123).randn(w_avg_samples, G.z_dim)).to(device)
w_samples = G.mapping(z_samples, c_samples)  # [N, L, C]

w = gen_utils.get_w_from_seed(G, batch_sz, device, truncation_psi, seed=seed,
                                centroids_path=centroids_path, class_idx=class_idx)

w = gen_utils.get_w_from_seed(G, batch_sz, device, truncation_psi, seed=seed,
                                centroids_path=centroids_path, class_idx=class_idx)
img = gen_utils.w_to_img(G, w, to_np=True)

# -------------------------------------------------------------------------




target_vector = None
for i, value_normalized in enumerate(values_normalized) :
    class_name = class_en[i]
    class_vector = np.load(f'class_vector/{class_name}.npy')

    if target_vector is None :
        target_vector = class_vector * value_normalized
    else :
        target_vector += class_vector * value_normalized

projected_w = target_vector.copy()
projected_w = torch.from_numpy(projected_w).to(device)
projected_w.shape # (1, 1, 512)

synth_image = G.synthesis(projected_w.repeat(1, G.num_ws, 1))
synth_image = (synth_image + 1) * (255/2)
synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
Image.fromarray(synth_image, 'RGB')