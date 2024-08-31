import random
import gradio as gr
from PIL import Image
import numpy as np
import torch

import sys
sys.path.append('./stylegan-xl')
import dnnlib, legacy

NETWORK = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda')
with dnnlib.util.open_url(NETWORK) as f:
    G = legacy.load_network_pkl(f)['G_ema']
    G = G.eval().requires_grad_(False).to(device)

def randomize_values(*args) :
    return [round(random.random(), 2) for _ in range(len(args))]

def generate_pokemon(normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy) :
    class_values = [normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy]
    if 0 in class_values :
        raise gr.Error('0 values are not allowed')
    
    
    return Image.new('RGB', (256,256), (255,128,128))

with gr.Blocks() as demo :
    with gr.Row() :
        with gr.Column() :
            with gr.Row() :
                normal = gr.Slider(label='Normal', minimum=0, maximum=1, value=0, step=0.01)
                fire = gr.Slider(label='Fire', minimum=0, maximum=1, value=0, step=0.01)
                water = gr.Slider(label='Water', minimum=0, maximum=1, value=0, step=0.01)
                grass = gr.Slider(label='Grass', minimum=0, maximum=1, value=0, step=0.01)
                electric = gr.Slider(label='Electric', minimum=0, maximum=1, value=0, step=0.01)
                ice = gr.Slider(label='Ice', minimum=0, maximum=1, value=0, step=0.01)
                fighting = gr.Slider(label='Fighting', minimum=0, maximum=1, value=0, step=0.01)
                poison = gr.Slider(label='Poison', minimum=0, maximum=1, value=0, step=0.01)
                ground = gr.Slider(label='Ground', minimum=0, maximum=1, value=0, step=0.01)
                flying = gr.Slider(label='Flying', minimum=0, maximum=1, value=0, step=0.01)
                psychic = gr.Slider(label='Psychic', minimum=0, maximum=1, value=0, step=0.01)
                bug = gr.Slider(label='Bug', minimum=0, maximum=1, value=0, step=0.01)
                rock = gr.Slider(label='Rock', minimum=0, maximum=1, value=0, step=0.01)
                ghost = gr.Slider(label='Ghost', minimum=0, maximum=1, value=0, step=0.01)
                dragon = gr.Slider(label='Dragon', minimum=0, maximum=1, value=0, step=0.01)
                dark = gr.Slider(label='Dark', minimum=0, maximum=1, value=0, step=0.01)
                steel = gr.Slider(label='Steel', minimum=0, maximum=1, value=0, step=0.01)
                fairy = gr.Slider(label='Fairy', minimum=0, maximum=1, value=0, step=0.01)

            with gr.Row() :
                btn_random = gr.Button('Random', variant='secondary')
                btn_generate = gr.Button('Generate', variant='primary')
        
        output_img = gr.Image(label='Output', image_mode='RGB', type='pil')

    btn_random.click(
        fn=randomize_values,
        inputs=[normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy],
        outputs=[normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy]
    )
    btn_generate.click(
        fn=generate_pokemon,
        inputs=[normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy],
        outputs=output_img,
        concurrency_id='gpu'
    )

demo.queue(default_concurrency_limit=1)
demo.launch(show_api=False)