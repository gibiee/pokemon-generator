import sys
sys.path.append('./stylegan-xl')

import os, glob, copy
import dill, imageio
import numpy as np
from PIL import Image
import torch
import dnnlib, legacy
from time import perf_counter
from tqdm import tqdm

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


vector_path = 'class_vector/Rock.npy'
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

w1_path = 'class_vector/Fire.npy'
w1 = np.load(w1_path)
w1 = torch.from_numpy(w1).to(device)

w2_path = 'class_vector/Poison.npy'
w2 = np.load(w2_path)
w2 = torch.from_numpy(w2).to(device)

w = w1 + w2
w = (w1 + w2) / 2
w = torch.mean()

synth_image = G.synthesis(w.repeat(1, G.num_ws, 1))
synth_image = (synth_image + 1) * (255/2)
synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
Image.fromarray(synth_image, 'RGB')

# 최종 구현 방향은 가중치를 각 클래스 벡터별로 곱해서 더하는 것이므로 mean()이 필요 없음.