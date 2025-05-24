import os
from openai import OpenAI
import subprocess
from prompts import prompt_user
from utils import append_to_json, filter_json, read_json
import json
import re
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from prompts import prompt_pattern
from config import Config


def run_llm(prompt: str) -> str:
    client = OpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_API_BASE
    )
    completion = client.chat.completions.create(
        model=Config.model_name,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return completion.choices[0].message.content


def process_item(item: Dict, error_path: str) -> Dict:
    """Function to process a single data item"""
    parsed_result = None
    result = None
    try:
        # Check if already processed
        if "pattern" in item and item["pattern"]:
            return item

        prompt = list(item.values())[0]
        formatted_prompt = prompt_user.format(prompt=prompt)
        result = run_llm(formatted_prompt)
        parsed_result = filter_json(result)
        item["result"] = parsed_result
        item["pattern"] = parsed_result.get("pattern", "")
        return item
    except Exception as e:
        error_data = {
            "prompt": prompt,
            "error": f"Processing failed: {str(e)}"
        }
        if result is not None:
            error_data["result"] = result
            print(result)
        print(f"An unexpected error occurred: {e}")
        if "Input data may contain inappropriate content" in str(e):
            item["is harmful"] = True
        return item


def process_dataset(
        dataset_path: str,
        error_path: str,
        max_workers: int = None,
        start_index: int = 0,
        end_index: int = None
):
    """Process the dataset in parallel, with support for start and end index"""
    # Read data
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Slice based on start and end index
    if end_index is None:
        end_index = len(data)

    data_slice = data[start_index:end_index]

    # Use process pool for parallel processing
    futures_map = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for local_idx, item in enumerate(data_slice):
            future = executor.submit(process_item, item, error_path)
            futures_map[future] = local_idx

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

            # Save progress periodically
            if processed_count % 10 == 0:
                with open(dataset_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
                print(f"Saved progress at item {start_index + local_idx + 1}")

    # Final save
    with open(dataset_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    max_workers = 5

    folder_path = ''
    error_path1 = ''
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            dataset_path_test = os.path.join(folder_path, filename)
            error_path = os.path.join(error_path1, f"{filename.split('.')[0]}.json")

            print(f"Processing file: {dataset_path_test}")

            # Call process_dataset function for each file
            process_dataset(
                dataset_path=dataset_path_test,
                error_path=error_path,
                max_workers=max_workers,
                start_index=0  # Start index
            )

            print(f"{dataset_path_test} completed")


if __name__ == "__main__":
    main()
