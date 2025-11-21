#!/usr/bin/env python3
"""
Script to download Qwen3-Reranker-8B model weights to local directory
Use this script to download model weights locally for faster Cogs deployment.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def download_model():
    model_id = 'Qwen/Qwen3-Reranker-8B'
    local_dir = './model_weights'

    print(f"Downloading model weights from {model_id}")
    print(f"Saving to: {local_dir}")

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    print('Downloading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(local_dir)

    print('Downloading model...')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto'
    )
    model.save_pretrained(local_dir)

    print(f'✓ Model weights saved to {local_dir}!')
    print(f'Contents of {local_dir}:')
    for item in os.listdir(local_dir):
        print(f'  - {item}')

if __name__ == "__main__":
    download_model()