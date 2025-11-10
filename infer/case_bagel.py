'''
Refer to https://github.com/ByteDance-Seed/Bagel/blob/main/inference.ipynb
'''
import os
import random
import numpy as np
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

from loader import WeaveBench, concatenate_image
from tqdm import tqdm
from PIL import Image
import tempfile

model_path = "./BAGEL-7B-MoT/"  # YOUR MODEL PATH

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

max_mem_per_gpu = "40GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
)

model = model.eval()
print('Model loaded')

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

img_hyper=dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)
txt_hyper=dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
)

############################## Begin our WeaveBench Test ##################################

SAVE_DIR = "./result/bagel/"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR+'/imgs', exist_ok=True)
ds = WeaveBench(incontext_mode='partial', modality_mode='unified', save_directory=SAVE_DIR)
for item in tqdm(ds):
    for idx, chat in enumerate(ds.iterate_chats_pairs(item['chats'])):
        raw_image_path = ds.get_image_path(chat[0]['content'], item['images'], True, idx, return_type=chat[0]['type'])
        save_image_path = ds.get_image_path(chat[1]['content'], item['images'], False, idx, return_type=chat[1]['type'])[0]
        assert SAVE_DIR in save_image_path
        instr = ds.refine_instruction(chat[0]['content'])

        if os.path.exists(save_image_path):
            try:
                if save_image_path.endswith('.txt'):
                    continue
                img = Image.open(save_image_path)
                img.verify()
                img.close()
                continue
            except:
                print('image demaged')
        
        input_image = [Image.open(x).convert("RGB") for x in raw_image_path]
        # for concate mode
        # if len(raw_image_path) == 1:
        #     input_image = [Image.open(raw_image_path[0]).convert("RGB")]
        # else:
        #     input_image = [concatenate_image([Image.open(x).convert("RGB") for x in raw_image_path])]

        # Image generation problem
        if chat[1]['type'] == 'image': 
            output_dict = inferencer.interleave_call(input_image + [instr], **img_hyper)
            output_dict['image'].save(save_image_path)
        
        # VQA problem
        else:
            output_dict = inferencer.interleave_call(input_image + [instr], understanding_output=True, **txt_hyper)
            with open(save_image_path, 'w', encoding='utf-8') as f:
                f.write(output_dict['text'])