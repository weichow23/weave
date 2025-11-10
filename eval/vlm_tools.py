import base64
import requests
import time
from io import BytesIO
from typing import Union, Optional, Tuple, List
from PIL import Image, ImageOps
from termcolor import cprint
import os
from config import OPENAI_MODEL, AZURE_API_KEY, AZURE_ENDPOINT

# Function to encode a PIL image
def encode_pil_image(pil_image):
    # Create an in-memory binary stream
    image_stream = BytesIO()
    
    # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
    pil_image.save(image_stream, format='JPEG')
    
    # Get the binary data from the stream and encode it as base64
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image


def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    """
    Load an image from a given path or URL and convert it to a PIL Image.

    Args:
        image (Union[str, Image.Image]): The image path, URL, or a PIL Image object to be loaded.
        format (str, optional): Desired color format of the resulting image. Defaults to "RGB".
        size (Optional[Tuple], optional): Desired size for resizing the image. Defaults to None.

    Returns:
        Image.Image: A PIL Image in the specified format and size.

    Raises:
        ValueError: If the provided image format is not recognized.
    """
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
        for attempt in range(self.max_retries):
            try:
                content = self.get_parsed_output(prompt)
                if content:
                    return content  # Return content for extract_score_and_reason to handle
                else:
                    print(f"Attempt {attempt + 1}: Empty response, retrying...")
                    
                time.sleep(2 ** min(attempt, 3))  # Exponential backoff, capped at 8 seconds
                
            except Exception as e:
                print(f"OpenAI GPT evaluation attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** min(attempt, 3))  # Exponential backoff

        print(f"OpenAI GPT evaluation failed after {self.max_retries} attempts.")
        return ""

def pack_prompt_with_path(prompt_text, img_list)->List:
    parts = prompt_text.split('<image>')
    result = []
    
    if len(parts) - 1 > len(img_list):
        raise ValueError(f"Not enough images provided. Need {len(parts)-1}, but got {len(img_list)}")
    
    for i in range(len(parts)):
        # Add text part if not empty
        if parts[i]:
            result.append(parts[i])
        # Add image path if available
        if i < len(parts) - 1 and i < len(img_list):
            result.append(img_list[i])
    
    return result

if __name__ == "__main__":
    # TEST YOUR GPT here
    model = GPT()
    prompt = model.prepare_prompt(['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg', 'What is difference between two images?'])
    print("prompt : \n", prompt)
    res = model(prompt)
    print("result : \n", res)