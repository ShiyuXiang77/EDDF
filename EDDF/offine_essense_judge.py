from prompts import judge_prompt
from utils import append_to_json, filter_json, read_json
import json
import os
import re
import subprocess
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompts import prompt_pattern
from utils import append_to_json, filter_json
import requests
from openai import OpenAI
from config import Config

def run_llm(prompt: str) -> str:
    client = OpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_API_BASE
    )
    completion = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return completion.choices[0].message.content

def process_item(item: Dict, error_path: str) -> Dict:
    """Process a single data item"""
    if "judge" in item:
        return item
    prompt = item["adversarial"]
    components = str(item['components'])
    pattern = str(item['essence pattern'])
    formatted_prompt = judge_prompt.format(jailbreak_prompt=prompt, components=components, pattern=pattern)
    try:
        result = run_llm(formatted_prompt)
        parsed_result = filter_json(result)
        item["judge"] = parsed_result
        return item
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        data1 = {
            "prompt": prompt,
            "error": f"Pattern parsing failed: {e}"
        }
        append_to_json(error_path, data1)
        return item

def process_dataset(
        dataset_path: str,
        error_path: str,
        max_workers: int = None,
        start_index: int = 0,
        end_index: int = None
):
    """Parallel processing of the dataset, supports start and end indices"""
    # Read data
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Slice based on start and end index
    if end_index is None:
        end_index = len(data)

    data_slice = data[start_index:end_index]

    # Process in parallel using a process pool
    futures_map = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for local_idx, item in enumerate(data_slice):
            future = executor.submit(process_item, item, error_path)
            futures_map[future] = local_idx

        # Process completed futures
        processed_count = 0
        for future in as_completed(futures_map.keys()):
            local_idx = futures_map[future]
            try:
                result = future.result()
                data[start_index + local_idx] = result
                print(f"Processed item {start_index + local_idx}")
                processed_count += 1
            except Exception as e:
                print(f"Error processing item {start_index + local_idx}: {e}")

            # Periodic save
            if processed_count % 20 == 0:
                with open(dataset_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
                print(f"Saved progress at item {start_index + local_idx + 1}")

    # Final save
    with open(dataset_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    max_workers = 6  # Reserve one CPU core

    folder_path = ''
    error_path1 = ''

    # Iterate over all JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            dataset_path_test = os.path.join(folder_path, filename)
            error_path = os.path.join(error_path1, f"{filename.split('.')[0]}.json")

            print(f"Processing file: {dataset_path_test}")

            # Call process_dataset to process each file
            process_dataset(
                dataset_path=dataset_path_test,
                error_path=error_path,
                max_workers=max_workers,
                start_index=0,  # Start index
                end_index=50
            )

            print(f"{dataset_path_test} completed")

if __name__ == '__main__':
    main()
