import base64
import requests
import time
from io import BytesIO
from typing import Union, Optional, Tuple, List
from PIL import Image, ImageOps
from termcolor import cprint

import os
from loader import WeaveBench, llm_yes_first, llm_yes_front, pack_prompt_with_path
from tqdm import tqdm
from PIL import Image
import tempfile


OPENAI_MODEL = <YOUR MODEL>
AZURE_API_KEY = <YOUR KEY>
AZURE_ENDPOINT = <YOUR ENDPOINT>

def encode_pil_image(pil_image):
    image_stream = BytesIO()
    pil_image.save(image_stream, format='JPEG')
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image


def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if (size != None):
        image = image.resize(size, Image.LANCZOS)
    return image

class GPT():
    def __init__(self, are_images_encoded=False, max_retries=3):
        """OpenAI GPT-4-vision model wrapper
        Args:
            are_images_encoded (bool): Whether the images are encoded in base64. Defaults to False.
        """
        self.use_encode = are_images_encoded
        self.max_retries = max_retries

    def prepare_prompt(self, content_list: List):
        """
        Automatically identify and process each element as image path or text
        
        Args:
            content_list (List): List containing text and image paths
        
        Returns:
            List: Formatted prompt content list for API
        """
        prompt_content = []
        
        # If input is not a list, convert to list
        if not isinstance(content_list, list):
            content_list = [content_list]
        
        for item in content_list:
            # Check if it's a string
            if isinstance(item, str):
                # Check if string might be an image path
                is_image_path = any([
                    item.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
                ]) or item.startswith(('http://', 'https://', 'data:image'))
                
                if is_image_path:
                    # Process image path
                    try:
                        image = load_image(item)
                        
                        # For local files or when self.use_encode is True, always use base64 encoding
                        if self.use_encode or os.path.isfile(item):
                            encoded_image = encode_pil_image(image)
                            visual_dict = {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                            }
                        else:
                            # For HTTP/HTTPS URLs when use_encode is False
                            visual_dict = {
                                "type": "image_url",
                                "image_url": {"url": item}
                            }
                        prompt_content.append(visual_dict)
                    except Exception as e:
                        print(f"Error processing image {item}: {e}")
                        # Skip this item if there's an error
                        continue
                else:
                    # Process text string
                    text_dict = {
                        "type": "text",
                        "text": item
                    }
                    prompt_content.append(text_dict)
            elif isinstance(item, Image.Image):
                # Handle PIL Image objects directly
                encoded_image = encode_pil_image(item)
                visual_dict = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                }
                prompt_content.append(visual_dict)
            else:
                # Skip if item is neither string nor PIL Image
                continue
        
        return prompt_content
    
    def get_parsed_output(self, prompt):
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
            {
                "role": "user",
                "content": prompt
            }
            ],
            "max_tokens": 1400
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AZURE_API_KEY}"
        }
        response = requests.post(AZURE_ENDPOINT, json=payload, headers=headers)
        return self.extract_response(response)
    
    def extract_response(self, response):
        response = response.json()

        try:
            out = response['choices'][0]['message']['content']
            return out
        except:
            if response['error']['code'] == 'content_policy_violation':
                print("Code is content_policy_violation")
            elif response['error']['code'] == 'rate_limit_exceeded' or response['error']['code'] == 'insufficient_quota':
                print(f"Code is {response['error']['code']}")
                print(response['error']['message'])
            else:
                print("Code is different")
                print(response)
        return ""

    def __call__(self, prompt) -> str:
        content = self.get_parsed_output(prompt)
        return content


if __name__ == "__main__":
    model = GPT()

    SAVE_DIR = "./result/gpt4.1/"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR+'/imgs', exist_ok=True)
    ds = WeaveBench(incontext_mode='yes', modality_mode='unified', save_directory=SAVE_DIR) # always keep no for VLM
    TEST_MODE = 'yes-front' # NOTE: choose it from 'yes-front' 'yes-firt' 'no'
    for item in tqdm(ds):
        history_chat = []
        for idx, chat in enumerate(ds.iterate_chats_pairs(item['chats'])):
            if chat[1]['type'] == 'image':
                history_chat+=[
                    {
                        "role": "user",
                        "content": chat[0]['content']
                    },
                    {
                        "role": "assistant",
                        "content": chat[1]['content']
                    }
                ]
                continue
            raw_image_path = ds.get_image_path(chat[0]['content'], item['images'], True, idx, return_type=chat[0]['type'])
            save_image_path = ds.get_image_path(chat[1]['content'], item['images'], False, idx, return_type=chat[1]['type'])[0]
            assert SAVE_DIR in save_image_path
            if os.path.exists(save_image_path):
                history_chat+=[
                    {
                        "role": "user",
                        "content": chat[0]['content']
                    },
                    {
                        "role": "assistant",
                        "content": chat[1]['content']
                    }
                ]
                continue
            
            if TEST_MODE == 'no':
                instr = ds.refine_instruction(chat[0]['content'], '<image>')
                prompt_text = pack_prompt_with_path(instr, raw_image_path)
                prompt = model.prepare_prompt(prompt_text)
                message = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            elif TEST_MODE == 'yes-front':
                history_chat.append(
                    {
                        "role": "user",
                        "content": chat[0]['content']
                    }
                )
                # Process the history to build the message list
                message = []
                image_refs = llm_yes_front(history_chat, ds.images_directory, item['images'])

                for i, turn in enumerate(history_chat):
                    role = turn['role']
                    content = turn['content']
                    
                    # For the first user message, add the image references at the beginning
                    if i == 0 and role == 'user' and image_refs:
                        content = image_refs + [content]
                    else:
                        content = [content]
                    
                    message.append({
                        "role": role,
                        "content": model.prepare_prompt(content)
                    })

            elif TEST_MODE == 'yes-first'
                history_chat.append(
                    {
                        "role": "user",
                        "content": chat[0]['content']
                    }
                )
                message = [] 
                for turn in history_chat:
                    content = turn['content']
                

                for turn in history_chat:
                    # Get the content from the turn
                    content = turn['content']
                    
                    modified_content = llm_yes_first(content, ds.images_directory, item['images'], turn['role'])
                    
                    message.append({
                        "role": turn['role'],
                        "content": model.prepare_prompt(modified_content)
                    })


            #### Begin Test 
            payload = {
                "model": OPENAI_MODEL,
                "messages": message,
                "max_tokens": 1024
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AZURE_API_KEY}"
            }
            response = requests.post(AZURE_ENDPOINT, json=payload, headers=headers)
            res = model.extract_response(response)
            if res:
                with open(save_image_path, 'w', encoding='utf-8') as f:
                    f.write(res)