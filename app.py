import random
import gradio as gr
from PIL import Image
import numpy as np
import torch

import sys
sys.path.append('./stylegan-xl')
import dnnlib, legacy
from torch_utils import gen_utils

NETWORK = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl"
CLASS_NAMES = ["Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]
CLASS_FEATURES = {cname: np.load(f'class_features/{cname}.npy') for cname in CLASS_NAMES}
device = torch.device('cuda')

with dnnlib.util.open_url(NETWORK) as f:
    G = legacy.load_network_pkl(f)['G_ema']
    G = G.eval().requires_grad_(False).to(device)

def reset_values(*args) :
    return [0 for _ in range(len(args))]
            
def randomize_values(*args) :
    return [round(random.random(), 2) for _ in range(len(args))]

def generate_base_pokemon() :
    random_seed = random.randint(0, 2**32 - 1)
    base_w = gen_utils.get_w_from_seed(G, batch_sz=1, device=device, seed=random_seed)
    output = gen_utils.w_to_img(G, base_w, to_np=True)[0]
    output_img = Image.fromarray(output, 'RGB')
    return (base_w, output_img), output_img, [output_img]

def apply_class_feature(base_state, base_ratio, normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy) :
    if base_state is None :
        raise gr.Error('Base Pokémon is required.')
    
    class_values = [normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy]
    if sum(class_values) == 0 :
        return base_state[-1]
    
    # class_values = np.array(class_values)
    target_feature = None
    for i, class_value in enumerate(class_values) :
        if class_value == 0 : continue

        class_name = CLASS_NAMES[i]
        class_feature = CLASS_FEATURES[class_name]

        if target_feature is None :
            target_feature = class_feature * class_value
        else :
            target_feature += class_feature * class_value
    
    target_feature = torch.from_numpy(target_feature).to(device)
    target_feature = target_feature.repeat(1, G.num_ws, 1)

    base_w = base_state[0]
    edit_w = (base_w * base_ratio) + (target_feature * (1 - base_ratio))
    output_edit = gen_utils.w_to_img(G, edit_w, to_np=True)[0]
    return Image.fromarray(output_edit, 'RGB')

def add_image_to_gallery(gallery, img) :
    gallery.append(img)
    return gallery

with gr.Blocks() as demo :
    gr.Markdown('# Pokémon Generator')
    gr.Markdown('Generate a base Pokémon. Then, apply class features to edit the base Pokémon.')
    with gr.Row() :
        with gr.Column() :
            btn_generate = gr.Button('Generate a base Pokémon', variant='primary')
            base_state = gr.State()

            with gr.Row() :
                normal = gr.Slider(label='Normal', minimum=-1, maximum=1, value=0, step=0.01)
                fire = gr.Slider(label='Fire', minimum=-1, maximum=1, value=0, step=0.01)
                water = gr.Slider(label='Water', minimum=-1, maximum=1, value=0, step=0.01)
                grass = gr.Slider(label='Grass', minimum=-1, maximum=1, value=0, step=0.01)
                electric = gr.Slider(label='Electric', minimum=-1, maximum=1, value=0, step=0.01)
                ice = gr.Slider(label='Ice', minimum=-1, maximum=1, value=0, step=0.01)
                fighting = gr.Slider(label='Fighting', minimum=-1, maximum=1, value=0, step=0.01)
                poison = gr.Slider(label='Poison', minimum=-1, maximum=1, value=0, step=0.01)
                ground = gr.Slider(label='Ground', minimum=-1, maximum=1, value=0, step=0.01)
                flying = gr.Slider(label='Flying', minimum=-1, maximum=1, value=0, step=0.01)
                psychic = gr.Slider(label='Psychic', minimum=-1, maximum=1, value=0, step=0.01)
                bug = gr.Slider(label='Bug', minimum=-1, maximum=1, value=0, step=0.01)
                rock = gr.Slider(label='Rock', minimum=-1, maximum=1, value=0, step=0.01)
                ghost = gr.Slider(label='Ghost', minimum=-1, maximum=1, value=0, step=0.01)
                dragon = gr.Slider(label='Dragon', minimum=-1, maximum=1, value=0, step=0.01)
                dark = gr.Slider(label='Dark', minimum=-1, maximum=1, value=0, step=0.01)
                steel = gr.Slider(label='Steel', minimum=-1, maximum=1, value=0, step=0.01)
                fairy = gr.Slider(label='Fairy', minimum=-1, maximum=1, value=0, step=0.01)

            with gr.Row() :
                btn_zero = gr.Button('Reset to 0', variant='secondary')
                btn_random = gr.Button('Randomize', variant='secondary')
                btn_edit = gr.Button('Apply class feature', variant='primary')
            
            base_ratio = gr.Slider(label='Base ratio', minimum=0, maximum=1, value=0.5, step=0.01,
                                   info="The parameter that determines how much the base Pokémon's form is retained.")
        
        with gr.Column() :
            output_img = gr.Image(label='Output', image_mode='RGB', type='pil', interactive=False)
            output_gallery = gr.Gallery(label='Gallery', columns=5, interactive=False)

    btn_zero.click(
        fn=reset_values,
        inputs=[normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy],
        outputs=[normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy]
    )

    btn_random.click(
        fn=randomize_values,
        inputs=[normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy],
        outputs=[normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy]
    )

    btn_generate.click(
        fn=generate_base_pokemon,
        inputs=None,
        outputs=[base_state, output_img, output_gallery],
        concurrency_id='gpu')
    
    btn_edit.click(
        fn=apply_class_feature,
        inputs=[base_state, base_ratio, normal, fire, water, grass, electric, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy],
        outputs=output_img,
        concurrency_id='gpu'
    ).success(
        fn=add_image_to_gallery,
        inputs=[output_gallery, output_img],
        outputs=output_gallery
    )

demo.title = 'Pokémon Generator'
demo.queue(default_concurrency_limit=1)
demo.launch(show_api=False)