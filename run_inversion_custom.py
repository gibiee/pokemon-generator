import sys
sys.path.append('./stylegan-xl')
# from run_inversion import get_morphed_w_code, space_regularizer_loss, pivotal_tuning
from run_inversion import pivotal_tuning, project

import os, glob
import dill, imageio
import numpy as np
import PIL.Image
import torch
import dnnlib, legacy
from time import perf_counter
from tqdm import tqdm

NETWORK = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon512.pkl"
# NETWORK = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl"
SAVE_DIR = 'projection'
SEED = 42
INV_STEPS = 100 # 1000
SAVE_VIDEO = True
PTI_STEPS = 35 # 350 # if 0 or None : do not pti
VERBOSE = False

target_paths = sorted(glob.glob('dataset/images/*.png'))
print(f'target_paths : {len(target_paths)}')
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)

# Load networks.
print('Loading networks from "%s"...' % NETWORK)
device = torch.device('cuda')
with dnnlib.util.open_url(NETWORK) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].to(device) # type: ignore

for target_path in tqdm(target_paths[:3]) :
    target_num, _ = os.path.splitext(os.path.basename(target_path))

    # Load target image.
    target_pil = PIL.Image.open(target_path).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Latent optimization
    print('Running Latent Optimization...')
    start_time = perf_counter()
    all_images = []
    all_images, projected_w = project(G, target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
                                      num_steps=INV_STEPS, device=device, verbose=VERBOSE, noise_mode='const',)
    print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')

    # Run PTI
    if PTI_STEPS not in [None, 0] :
        print('Running Pivotal Tuning Inversion...')
        start_time = perf_counter()

        # 여기서 G 갱신하지말고 처음 정의한 G로 테스트해야할듯.
        gen_images, G = pivotal_tuning(G, projected_w,
                                       target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
                                       device=device,
                                       num_steps=PTI_STEPS,
                                       verbose=VERBOSE)
        all_images += gen_images
        print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    if SAVE_VIDEO :
        video = imageio.get_writer(f'{SAVE_DIR}/{target_num}_proj.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{target_num}_proj.mp4"')
        for synth_image in all_images:
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{SAVE_DIR}/{target_num}_target.png')
    synth_image = G.synthesis(projected_w.repeat(1, G.num_ws, 1))
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{SAVE_DIR}/{target_num}_proj.png')

    # save latents
    np.savez(f'{SAVE_DIR}/{target_num}_projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    # Save Generator weights
    snapshot_data = {'G': G, 'G_ema': G}
    with open(f"{SAVE_DIR}/{target_num}_G.pkl", 'wb') as f:
        dill.dump(snapshot_data, f)