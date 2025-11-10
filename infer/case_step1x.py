import torch
from diffusers import Step1XEditPipelineV1P2

import os
from .loader import WeaveBench, concatenate_image
from tqdm import tqdm
from PIL import Image

pipe = Step1XEditPipelineV1P2.from_pretrained("stepfun-ai/Step1X-Edit-v1p2-preview", torch_dtype=torch.bfloat16)
pipe.to("cuda")
enable_thinking_mode=False
enable_reflection_mode=False

SAVE_DIR = "./result/step1x-v1.2/" # NOTE: need change
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR+'/imgs', exist_ok=True)
ds = WeaveBench(incontext_mode='partial', modality_mode='img', save_directory=SAVE_DIR) # NOTE: the mode is import 
for item in tqdm(ds):
    for chat in ds.iterate_chats_pairs(item['chats']):
        raw_image_path = ds.get_image_path(chat[0]['content'], item['images'], True)
        save_image_path = ds.get_image_path(chat[1]['content'], item['images'], False)[0]
        assert SAVE_DIR in save_image_path
        instr = ds.refine_instruction(chat[0]['content'])

        if os.path.exists(save_image_path):
            try:
                img = Image.open(save_image_path)
                img.verify()
                img.close()
                continue
            except:
                print('image demaged')
                
        
        if len(raw_image_path) == 1:
            input_image = Image.open(raw_image_path[0]).convert("RGB")
        else:
            input_image = concatenate_image([Image.open(x).convert("RGB") for x in raw_image_path])

        pipe_output = pipe(
            image=input_image,
            prompt=instr, 
            num_inference_steps=28,
            true_cfg_scale=4,
            generator=torch.Generator().manual_seed(42),
            enable_thinking_mode=enable_thinking_mode,
            enable_reflection_mode=enable_reflection_mode,
        )

        output_image = pipe_output.images[0]
        output_image.save(save_image_path)