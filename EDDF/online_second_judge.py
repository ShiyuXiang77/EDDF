import json
import shutil
import pandas as pd
from config import Config
from utils import append_to_json, read_json, filter_json
from prompts import second_judge, LLM_judge, prompt_pattern
from openai import OpenAI
import os
import re
import subprocess
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests

threshold = 0.5

def run_llm(prompt: str) -> str:
    client = OpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_API_BASE
    )
    completion = client.chat.completions.create(
        model=Config.model_name,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return completion.choices[0].message.content

def process_item(item: Dict, error_path: str) -> Dict:
    """Function to process a single data item"""
    result = None
    try:
        prompt_final_test = list(item.values())[0]
        pattern = item["pattern"]
        prompt_final_test = item.get("adversarial", "")
        similar_scores = item["scores"]
        similar_prompts = item['similar_prompt']
        similar_patterns = item['similar_pattern']
        # Create copies to avoid modification
        similar_prompts = similar_prompts.copy()
        similar_patterns = similar_patterns.copy()

        if similar_scores[0] > threshold:
            formatted_prompt = LLM_judge.format(prompt_user=prompt_final_test, query_essence=pattern)
            result = run_llm(formatted_prompt)
            parsed_result = filter_json(result)
            is_harmful = parsed_result.get("is_harmful", False)
            reasoning = parsed_result.get("reasoning", "")
            true_problem = parsed_result["true_problem"] if "true_problem" in parsed_result else ""
            item['true_problem'] = true_problem
            item['second result'] = result
            item['judge'] = True
            item["is harmful"] = is_harmful
            item["reasoning"] = reasoning
            item["second judge"] = False
            return item
        else:
            # Traverse similar_scores, and clear corresponding similar_prompt and similar_pattern if above threshold
            for a in range(len(similar_scores)):
                if similar_scores[a] > threshold:
                    similar_prompts[a] = ""
                    similar_patterns[a] = ""
            formatted_prompt = second_judge.format(
                prompt_user=prompt_final_test,
                query_essence=pattern,
                p1=similar_prompts[0], e1=similar_patterns[0],
                p2=similar_prompts[1], e2=similar_patterns[1],
                p3=similar_prompts[2], e3=similar_patterns[2],
                p4=similar_prompts[3], e4=similar_patterns[3],
                p5=similar_prompts[4], e5=similar_patterns[4]
            )
            result = run_llm(formatted_prompt)
            parsed_result = filter_json(result)
            is_harmful = parsed_result.get("is_harmful", False)
            reasoning = parsed_result.get("reasoning", "")
            true_problem = parsed_result["true_problem"] if "true_problem" in parsed_result else ""
            item['second result'] = result
            item['judge'] = True
            item['true_problem'] = true_problem
            item["is harmful"] = is_harmful
            item["reasoning"] = reasoning
            item["second judge"] = True
            return item
    except Exception as e:
        error_data = {
            "prompt": prompt_final_test,
            "error": f"Processing failed: {str(e)}"
        }
        if result is not None:
            error_data["result"] = result
            print(result)
        print(f"An unexpected error occurred: {e}")
        if "Input data may contain inappropriate content" in str(e):
            item['judge'] = True
            item["is harmful"] = True
            item["reasoning"] = "Input data may contain inappropriate content"
            item["second judge"] = True
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
