import json
import shutil
import pandas as pd
from vectorstore import VectorStore
from config import Config
from embedding import get_embedding_model
from utils import filter_json
import re
import os


def process_json_files(input_folder: str, k, output_folder):
    """
    Process all JSON files in a folder

    Args:
        input_folder (str): Path to the folder containing JSON files
    """
    # Ensure the folder exists
    if not os.path.exists(input_folder):
        print(f"Folder does not exist: {input_folder}")
        return
    # Create output_folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize VectorStore and embedding model
    vectorstore = VectorStore()
    model_name = Config.EMBEDDING_MODEL_NAME
    embedding_model = get_embedding_model(model_name)

    # Traverse all JSON files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                # Read the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Flag to indicate whether the file is modified
                # file_modified = False
                num = 0
                # Iterate over each data entry
                for item in data:
                    if item.get("similar_prompt"):
                        continue
                    # Check if 'pattern' exists and is non-empty
                    if item.get('pattern'):
                        try:
                            # Get the prompt (assume the first value is the prompt)
                            prompt_final_test = list(item.values())[0]
                            pattern = item['pattern']

                            # Generate query embedding
                            # query_embedding = embedding_model.embed_query(pattern)

                            # Perform similarity search
                            results = vectorstore.similarity_search(pattern, k)
                            num += 1
                            print(num)
                            # Process results
                            similar_prompts = []
                            similar_patterns = []
                            scores = []

                            for result, score in results:
                                similar_patterns.append(result.page_content)
                                similar_prompt = result.metadata.get("prompt", "")
                                similar_prompts.append(similar_prompt)
                                scores.append(score)

                            # Update data item
                            item['scores'] = scores
                            item['similar_pattern'] = similar_patterns
                            item['similar_prompt'] = similar_prompts

                            # Mark file as modified
                            file_modified = True

                        except Exception as e:
                            print(f"Error processing an item in {filename}: {e}")

                # If the file is modified, save it
                # if file_modified:
                with open(output_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
                    print(f"Processed and saved: {output_path}")

            except json.JSONDecodeError:
                print(f"Failed to parse JSON file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")


def main():
    # Specify the path to the folder to be processed
    input_folder = ''
    output_folder = ""
    k = 5
    # Call the processing function
    process_json_files(input_folder, k, output_folder)

if __name__ == "__main__":
    main()
