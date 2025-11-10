# Copyright (c) 2025 WEAVE Team
# SPDX-License-Identifier: Apache-2.0

import os
import re
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from termcolor import cprint
from prompts import (
    prompt_keypoint_checklist,
    prompt_visual_consistency,
    prompt_image_quality,
    prompt_txt_acc
)
from utils import (
    load_txt,
    parse_json,
    find_image_pattern_idx, find_gt_pattern_idx
)
from threading import Lock
from vlm_tools import GPT, pack_prompt_with_path
from config import WEAVE_METRICS, DATASET_PATH

file_lock = Lock()
def save_result_jsonl(result, output_jsonl_path):
    """Save evaluation result to JSONL file with thread safety"""
    with file_lock:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def process_task_evaluation(model, task, output_jsonl_path, mode, input_dir):
    """Process single task evaluation"""
    results = {'idx': task["idx"], 'domain': task['domain']}
    try:
        for i in range(0, len(task['chats']), 2):
            user = task['chats'][i]
            assitant = task['chats'][i+1]
            assert assitant['role'] == "assistant", 'Dataset JSON error: assistant role is not "assistant"'
            keypoints = assitant['key_point']

            results[f'turn {i//2}'] = {}
            if assitant["type"] == 'image' and mode in ['img', 'umm']:
                gen_img_idx = find_image_pattern_idx(assitant['content'])[0] - 1
                gen_img_path = os.path.join(input_dir, task['images'][gen_img_idx].split('/')[-1])

                kp_imgs_path = []
                for k in keypoints.keys():
                    if 'GT' in k:
                        kp_img_id = find_gt_pattern_idx(k)[0] - 1
                        kp_imgs_path.append(os.path.join(DATASET_PATH, task['images'][kp_img_id]))
                    elif 'Image' in k:
                        kp_img_id = find_image_pattern_idx(k)[0] - 1
                        kp_imgs_path.append(os.path.join(input_dir, task['images'][kp_img_id].split('/')[-1]))
                    else:
                        continue
                
                for metric in WEAVE_METRICS['img']:
                    if metric == "key_point":
                        requirments = "\n".join([f"{i+1}. {v}" for i, v in enumerate(keypoints.values())])
                        prompt = prompt_keypoint_checklist + requirments
                        prompt_text = pack_prompt_with_path(prompt, [gen_img_path]+kp_imgs_path)
                        resp = model(model.prepare_prompt(prompt_text))
                        resp_dict = parse_json(resp)
                    elif metric == "visual_consistency":
                        requirments = "\n".join([f"{i+1}. {v}" for i, v in enumerate(keypoints.values())])
                        prompt = prompt_visual_consistency + requirments
                        prompt_text = pack_prompt_with_path(prompt, [gen_img_path]+kp_imgs_path)
                        resp = model(model.prepare_prompt(prompt_text))
                        resp_dict = parse_json(resp)
                    elif metric == "image_quality":
                        prompt_text = pack_prompt_with_path(prompt_image_quality, [gen_img_path])
                        resp = model(model.prepare_prompt(prompt_text))
                        resp_dict = parse_json(resp) # dict
                    else:
                        return False
                    
                    results[f'turn {i//2}'][metric] = resp_dict
            elif assitant["type"] == 'text' and mode in ['txt', 'umm']:
                txt_path = re.sub(r'\.(jpg|png|jpeg)$', f'_{i//2}.txt', task['images'][0], flags=re.IGNORECASE)
                generated_answer = load_txt(os.path.join(input_dir,  txt_path.split('/')[-1]))
                for metric in WEAVE_METRICS['txt']:
                    if metric == "accuracy":
                        prompt = prompt_txt_acc.format(standard_answer=assitant['content'], generated_answer=generated_answer)
                        resp = model(model.prepare_prompt([prompt]))
                        resp_dict = parse_json(resp)
                        results[f'turn {i//2}'][metric] =resp_dict
            else:
                continue
        
        save_result_jsonl(results, output_jsonl_path)
    except Exception as e:
        cprint(f"Error processing task {task['idx']}: {str(e)}", 'red')
        return False
    
    return True

def run_evaluation(
    input_dir="rover_results",
    output_dir="rover_results",
    mode="umm",
    num_workers=10,
):
    """
    Run WEAVE evaluation using Hugging Face dataset
    
    Args:
        output_dir: Directory to save results
        num_workers: Number of parallel workers
    """
    model = GPT()

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl_path = os.path.join(output_dir, "metrics.jsonl")

    with open(os.path.join(DATASET_PATH,"test.json"), 'r', encoding='utf-8') as f:
        ds = json.load(f)
    
    # Load already evaluated tasks
    already_ds = set()
    if os.path.exists(output_jsonl_path):
        valid_ds = []
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    already_ds.add(data['idx'])
                except json.JSONDecodeError:
                    continue
        for item in ds:
            if item['idx'] not in already_ds:
                valid_ds.append(item)
        print(f"Skipped {len(already_ds)} already evaluated tasks")
    else:
        valid_ds = ds

    if not valid_ds:
        print("No tasks with generated images found. Please run generation first.")
        return
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for task in valid_ds:
            future = executor.submit(
                process_task_evaluation,
                model, task, output_jsonl_path, mode, input_dir
            )
            futures.append(future)
        
        # Process results with progress bar
        successful = 0
        failed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating WEAVE"):
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Future failed: {e}")
                failed += 1
    
    print(f"Evaluation completed: {successful} successful, {failed} failed")
    print(f"Results saved to: {output_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WEAVE Evaluation")
    parser.add_argument("--input_dir", type=str, default="rover_results", help="Output directory")
    parser.add_argument("--output_dir", type=str, default="rover_results", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--mode", type=str, default="umm", choices=["umm", "img", "txt"], help="Evaluation mode (umm, img, txt)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    run_evaluation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.workers,
        mode=args.mode
    )
