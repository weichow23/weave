#!/usr/bin/env python3
"""
Configuration file for WEAVE evaluation
"""

import os

ROOT_PATH = <YOUR_DATA_PATH>
OPENAI_MODEL =  <MODEL> # like 'gpt-4o-2024-08-06'
AZURE_API_KEY = <YOUR KEY>
AZURE_ENDPOINT = <YOUR ENDPOINT> # like "https://api.openai.com/v1/chat/completions"

AZURE_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", 
    "2024-08-01-preview"
)

# Evaluation settings
MAX_RETRIES = int(os.getenv(
    "MAX_RETRIES",
    "3"
))

############ WEAVE meta information
WEAVE_DOMAIN = {"Science": ["optics", "geography", "astronomy", "chemistry", "physics", "biology"],
                "Creation": ["edit", "recall", "fusion", "story"], 
                "Logic": ["spatial", "mathematics"], 
                "Game": ["chess_game", "visual_jigsaw", "minecraft", "maze"]}
WEAVE_METRICS = {"img": ["key_point", "visual_consistency", "image_quality"], "txt": ["accuracy"]}
METRIC_DISPLAY_NAMES = {
    "visual_consistency": "VC",
    "image_quality": "IQ",
    "average": "AVG"
}

