CUDA_VISIBLE_DEVICES=1 python gen_images.py \
    --network=training-runs/00005-stylegan3-r-dataset-gpus1-batch4-gamma8/network-snapshot-012000.pkl \
    --seeds=1-9 --outdir=test --trunc=0.5



import sys
sys.path.append('./stylegan3')

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import pickle
import torch
from PIL import Image

checkpoint_path = "/home/oscar/source/TOY/pokemon-generator/stylegan3/training-runs/00005-stylegan3-r-dataset-gpus1-batch4-gamma8/network-snapshot-012000.pkl"

with open(checkpoint_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

z = torch.randn([1, G.z_dim]).cuda()
c = None


output = G(z, c, noise_mode='const')

output.min()
output.max()

c = torch.zeros([1, G.c_dim]).cuda()



w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
img = G.synthesis(w, noise_mode='const', force_fp32=True)





c = [0] * 11
c = torch.zeros([1, G.c_dim]).cuda()



Image.fromarray(output[0].cpu().numpy(), 'RGB')


output.permute(0, 2, 3, 1)[0].min()

output[0].cpu().numpy()

t = (output.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
Image.fromarray(t[0].cpu().numpy())

# truncation_psi=truncation_psi, noise_mode=noise_mode
w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
img = G.synthesis(w, noise_mode='const', force_fp32=True)



Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()

        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
