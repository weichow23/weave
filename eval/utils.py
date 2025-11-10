# Copyright (c) 2025 WEAVE Team
# SPDX-License-Identifier: Apache-2.0
import json
import re
import threading

# Thread-safe file writing lock
lock = threading.Lock()

def find_image_pattern_idx(content):
    pattern = r'Image #(\d+)'
    matches = re.findall(pattern, content)
    return [int(x) for x in matches]

def find_gt_pattern_idx(content):
    pattern = r'GT #(\d+)'
    matches = re.findall(pattern, content)
    return [int(x) for x in matches]

def save_result_jsonl(result, key, output_jsonl_path):
    """Save evaluation result to JSONL file with thread lock"""
    with lock:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            data = {"key": key, "result": result}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def parse_json(response):
    response = response.replace("```json","")
    response = response.replace("```","")
    return json.loads(response)


def load_txt(file_path):
    try:
        # Open the file in read mode with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read the entire content of the file
            text_content = f.read()
        return text_content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None
