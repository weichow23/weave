import re
import os
import json
import numpy as np
from tqdm import tqdm
from typing import List
from termcolor import cprint
from PIL import Image, ImageDraw, ImageFont

ROOT_PATH = <YOUR_DATA_PATH> # NOTE: place this

def llm_yes_front(history_chat, images_dir, images):
    # Dictionary to store all unique image references
    image_references = {}
    
    # First pass: collect all image references from the entire history
    for turn in history_chat:
        content = turn['content']
        # Find all image placeholders
        for match in re.finditer(r"Image #(\d+)", content):
            idx = int(match.group(1))
            # Store reference if not already collected
            if idx not in image_references and idx <= len(images):
                image_references[idx] = f"{images_dir}/{images[idx-1]}"
    
    # Generate the image reference string for the first user message
    if image_references:
        image_refs = [[f"Image #{idx}:", path] for idx, path in sorted(image_references.items())]
        flat_list = [item for sublist in image_refs for item in sublist]
        return flat_list
    return []

def llm_yes_first(content, images_dir, images, role):
    # Initialize result list and position tracker
    result_list = []
    last_end = 0
    replaced_first = False
    
    # Find all image placeholder patterns
    for match in re.finditer(r"Image #(\d+)", content):
        idx = int(match.group(1))
        start, end = match.span()
        
        # Add text before the current match
        if start > last_end:
            result_list.append(content[last_end:start])
        
        # Add the image placeholder or replacement
        if role=='assistant':
            continue
        if not replaced_first:
            # For the first occurrence, add the image path separately
            result_list.append(f"Image #{idx}")
            result_list.append(f"{images_dir}/{images[idx-1]}")
            replaced_first = True
        else:
            # For other occurrences, keep as is
            result_list.append(match.group(0))
        
        last_end = end
    
    # Add remaining text after the last match
    if last_end < len(content):
        result_list.append(content[last_end:])
    
    return result_list

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

def find_image_pattern_idx(content):
    pattern = r'Image #(\d+)'
    matches = re.findall(pattern, content)
    return [int(x) for x in matches]

def get_contrasting_color(image, x, y, width, height):
	"""
	Determine a contrasting color (black or white) based on the average color of a specified area in the image.
    refer to https://github.com/USC-GVL/PhysBench/blob/main/eval/eval_utils/task_evaluator.py#L396
	"""
	# Crop the relevant part of the image
	cropped_image = image.crop((x, y, x + width, y + height))
	# Convert to numpy array for analysis
	np_image = np.array(cropped_image)
	# Calculate the average color
	average_color = np.mean(np_image, axis=(0, 1))
	# Brightness calculation based on perceived luminance
	brightness = np.sqrt(0.299 * average_color[0] ** 2 + 0.587 * average_color[1] ** 2 + 0.114 * average_color[2] ** 2)
	# Return white for dark backgrounds and black for light backgrounds
	return 'white' if brightness < 128 else 'black'

def resize_if_needed(combined_image, max_size=4096):
    width, height = combined_image.size
    max_dimension = max(width, height)
    
    if max_dimension > max_size:
        scale = max_size / max_dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        combined_image = combined_image.resize((new_width, new_height), Image.LANCZOS)
    
    return combined_image

def concatenate_image(images:List[Image.Image], rows:int=1, columns=None, separator_width=5, max_size=4096*2)->Image.Image:
    '''
    refer to https://github.com/USC-GVL/PhysBench/blob/main/eval/eval_utils/task_evaluator.py#L297

    Usage:
        combined_image = concatenate_image(imgs)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
            combined_image.save(tmp, format='JPEG')
            tmp.flush()
            img_path = tmp.name
    ''' 
	# Ensure we have the exact number of images needed
    if rows == 1:
        columns = len(images)
    elif len(images) != rows * columns:
        raise ValueError(f"Expected {rows * columns} images, but got {len(images)}.")

    # # Calculate the max width and height of images to standardize sizes
    # max_width = max(img.width for img in images)
    # max_height = max(img.height for img in images)
    # # Resize images to the max width and height
    # resized_images = [img.resize((max_width, max_height), Image.Resampling.LANCZOS) for img in images]
    # # Calculate the total width and height for the combined image
    # total_width = max_width * columns + separator_width * (columns - 1)
    # total_height = max_height * rows + separator_width * (rows - 1)
    
    # Find the maximum height and scale each image proportionally to that height. We don't need to ensure that every image is exactly the same.
    max_height = max(img.height for img in images)
    resized_images = []
    for img in images:
        scale = max_height / img.height
        new_width = int(img.width * scale)
        resized_images.append(img.resize((new_width, max_height), Image.Resampling.LANCZOS))
    row_widths = []
    for row in range(rows):
        row_width = sum(resized_images[row * columns + col].width for col in range(columns))
        row_width += separator_width * (columns - 1)
        row_widths.append(row_width)

    total_width = max(row_widths)
    total_height = max_height * rows + separator_width * (rows - 1)
    combined_image = Image.new('RGB', (total_width, total_height), color='white')
    # --------------

    # Place images in the specified grid
    x_offset = 0
    y_offset = 0
    for i, img in enumerate(resized_images):
        combined_image.paste(img, (x_offset, y_offset))
        if (i + 1) % columns == 0:  # Move to the next row after the last column
            x_offset = 0
            y_offset += img.height + separator_width
        else:  # Move to the next column
            x_offset += img.width + separator_width

    # Add numbers to each image for identification
    draw = ImageDraw.Draw(combined_image)
    try:
        # The font size should be determined based on the maximum height; otherwise, the font will appear too small.
        # font_size = (max_width + max_height) // 2 // 12
        font_size = max_height // 12
        font = ImageFont.load_default(size=font_size)
    except IOError:
        font = ImageFont.truetype("arial", 20)

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(resized_images):
        text = str(i + 1)
        text_x = x_offset + 10
        text_y = y_offset + 10
        text_width, text_height = font_size, font_size
        font_color = get_contrasting_color(combined_image, text_x, text_y, text_width, text_height)
        draw.text((text_x, text_y), text, fill=font_color, font=font)
        if (i + 1) % columns == 0:
            x_offset = 0
            y_offset += img.height + separator_width
        else:
            x_offset += img.width + separator_width

    return resize_if_needed(combined_image, max_size)

def extract_image_numbers(data):
    pattern = r'Image #(\d+)'
    matches = re.findall(pattern, data)
    idxs = list(set([int(num) for num in matches]))

    return sorted(idxs)

class WeaveBench:
    def __init__(self, 
            incontext_mode='yes', 
            modality_mode='unified', 
            save_directory=None,
            json_file_path=f"{ROOT_PATH}/test/test.json", 
            images_directory=f"{ROOT_PATH}/test",
        ):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.ds = json.load(f)
        print(f"✅ Load JSON: {json_file_path}")
        print(f"📊 Find {len(self.ds)} items")

        assert incontext_mode in ['yes', 'no', 'partial'], f'Not support {incontext_mode}'
        assert modality_mode in ['img','txt', 'unified'], f'Not support {modality_mode}'
        self.incontext_mode = incontext_mode
        self.modality_mode = modality_mode

        for idx, item in enumerate(self.ds):
            new_chats = []
            for user_msg, assistant_msg in self.iterate_chats_pairs(item['chats']):
                task_mode = self.detect_mode([user_msg, assistant_msg])
                user_msg['task_mode'] = task_mode
                assistant_msg['task_mode'] = task_mode

                if self.modality_mode=='txt' and task_mode=='vqa':
                    new_chats+=[user_msg, assistant_msg]
                elif self.modality_mode=='img' and task_mode!='vqa':
                    new_chats+=[user_msg, assistant_msg]
                elif self.modality_mode=='unified':
                    new_chats+=[user_msg, assistant_msg]
                
            item['chats'] = new_chats

        self.save_directory = save_directory
        self.images_directory = images_directory
        self.img_idxs = None
    
    def __iter__(self):
        return iter(self.ds)
            
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return self.ds[idx]
    
    def iterate_chats_pairs(self, chats):
        """Each iteration returns two consecutive chats."""
        for i in range(0, len(chats), 2):
            if i + 1 < len(chats):
                yield chats[i], chats[i + 1]
            else:
                yield chats[i], None

    def detect_mode(self, chat):
        num_user = chat[0]['content'].count("Image #")
        type_assitant = chat[1]['type']
        if type_assitant == 'image':
            if num_user == 0:
                return 't2i'
            elif num_user == 1:
                return 'edit'
            else:
                return 'fusion'
        else:
            return 'vqa'
        
    def refine_instruction(self, text, new_pattern=None):
        '''
        Replace the Image # numbers with a consecutive sequence starting from 1, maintaining the size relationship. For example, if the text contains Image #5, Image #3, and Image #1, change them to Image #3, Image #2, and Image #1.
        Specifically, if there is only one Image #, change it to "image".

        Args:

        text: Input text

        new_pattern: New replacement pattern, can contain {num} placeholders

        For example: "<image>", "<image> {num}", "[IMG_{num}]"

        If None, use the default "Image #X" format

        # Example text

        text = "See Image #5, Image #3, and Image #1"

        # Default behavior (maintain original logic)

        result1 = refine_instruction(text)

        # Output: "See Image #3, Image #2, and Image #1"

        # Replace with a uniform <image>

        result2 = refine_instruction(text, new_pattern="<image>")

        # Output: "See <image>, <image>, and <image>"

        # Replace with numbered <image>

        result3 = refine_instruction(text, new_pattern="<image> {num}")

        # Output: "See <image> 3 and <image> 2" Also <image> 1"

        # Custom format

        result4 = refine_instruction(text, new_pattern="[IMG_{num}]")

        # Output: "See [IMG_3], [IMG_2], and [IMG_1]"
        '''

        if self.incontext_mode == 'yes':
            return text

        pattern = r'Image #(\d+)'
        matches = re.findall(pattern, text)
        if not matches:
            return text
        
        # If only one image
        if len(matches) == 1:
            if new_pattern is not None:
                # If new_pattern contains {num}, replace it with 1; otherwise, use it directly.
                if '{num}' in new_pattern:
                    return re.sub(pattern, new_pattern.replace('{num}', '1'), text)
                else:
                    return re.sub(pattern, new_pattern, text)
            else:
                return re.sub(pattern, 'image', text)
        
        # Get unique image numbers and sort them
        unique_nums = sorted(set(int(m) for m in matches))
        num_mapping = {old: idx + 1 for idx, old in enumerate(unique_nums)}
        
        # Replace each occurrence with new number
        def replace_func(match):
            old_num = int(match.group(1))
            new_num = num_mapping[old_num]
            
            if new_pattern is not None:
                # Replace the placeholder if new_pattern contains {num}.
                if '{num}' in new_pattern:
                    return new_pattern.replace('{num}', str(new_num))
                else:
                    # If no placeholders are used, all images will use the same pattern.
                    return new_pattern
            else:
                return f'Image #{new_num}'
        
        result = re.sub(pattern, replace_func, text)
        return result
    
    def set_history(self, history_chat):
        self.img_idxs = extract_image_numbers(history_chat)
        
    def get_image_path(self, content, imgs_path_list, input_flag=False, turn_idx=None, return_type='image')->List[str]:
        img_idxs = find_image_pattern_idx(content)
        img_paths = [imgs_path_list[img_idx-1] for img_idx in img_idxs]

        if input_flag: # Returns the path to the input field
            if self.incontext_mode=='no':
                return [os.path.join(self.images_directory, pt) for pt in img_paths]
            elif self.incontext_mode in ['yes', 'partial']:
                if self.incontext_mode=='yes': # In yes mode, to see all the images above, you need to use max to prevent incorrect numbering.
                    img_paths = [imgs_path_list[img_idx-1] for img_idx in range(max(img_idxs)+1)]
                
                full_paths = []
                for pt in img_paths:
                    save_path = os.path.join(self.save_directory, pt)
                    if os.path.exists(save_path):
                        full_paths.append(save_path)  # If it exists, use the path from save_directory.
                    else:
                        full_paths.append(os.path.join(self.images_directory, pt))  # If it does not exist, use images_directory
                return full_paths
            else:
                raise ValueError(f'Not support {self.incontext_mode}')           

        else: # Return output path
            if return_type=='image':  # Image saving
                return [os.path.join(self.save_directory, pt) for pt in img_paths]
            elif return_type=='text': # Save Text
                img_path = imgs_path_list[0]
                if turn_idx is None:
                    raise ValueError('QA need turn_idx input')
                txt_path = re.sub(r'\.(jpg|png|jpeg)$', f'_{turn_idx}.txt', img_path, flags=re.IGNORECASE)
                return [os.path.join(self.save_directory, txt_path)]
            else:
                raise ValueError(f'No type {return_type}')

if __name__ == "__main__":
    # A Test Case
    SAVE_DIR = ''
    ds = WeaveBench(incontext_mode='partial', modality_mode='img', save_directory=SAVE_DIR)
    for item in tqdm(ds[:10]):
        for idx, chat in enumerate(ds.iterate_chats_pairs(item['chats'])):
            raw_image_path = ds.get_image_path(chat[0]['content'], item['images'], True)
            save_image_path = ds.get_image_path(chat[1]['content'], item['images'], False)[0]
            assert SAVE_DIR in save_image_path
            instr = ds.refine_instruction(chat[0]['content'])

            break
        break
