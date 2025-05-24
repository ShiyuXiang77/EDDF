import os
import multiprocessing
# Import processing functions from each module
from online_user_essence import process_dataset as essence_process_dataset
from online_user_match import process_json_files
from online_second_judge import process_dataset as judge_process_dataset


def main():
    """Main function - Execute three processing steps in sequence"""

    # Configuration parameters
    max_workers = 5  # Reserve one CPU core
    folder_path = ''  # Path to the input folder
    error_path1 = ''  # Path to output error logs
    k = 5  # Matching parameter

    print("Starting data processing workflow...")

    # ========== Step 1: User essence feature extraction ==========
    print("\n=== Step 1: Performing user essence feature extraction ===")

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            dataset_path_test = os.path.join(folder_path, filename)
            error_path = os.path.join(error_path1, f"{filename.split('.')[0]}.json")

            print(f"Processing file: {dataset_path_test}")

            # Call user essence feature extraction function
            essence_process_dataset(
                dataset_path=dataset_path_test,
                error_path=error_path,
                max_workers=max_workers,
                start_index=0
            )

            print(f"{dataset_path_test} completed")

    print("Step 1 completed!")

    # ========== Step 2: User matching processing ==========
    print("\n=== Step 2: Performing user matching processing ===")

    input_folder = folder_path  # Use the same folder
    output_folder = folder_path  # Same as input_folder

    # Call user matching processing function
    process_json_files(input_folder, k, output_folder)

    print("Step 2 completed!")

    # ========== Step 3: Secondary judgment processing ==========
    print("\n=== Step 3: Performing secondary judgment processing ===")

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            dataset_path_test = os.path.join(folder_path, filename)
            error_path = os.path.join(error_path1, f"{filename.split('.')[0]}.json")

            print(f"Processing file: {dataset_path_test}")

            # Call secondary judgment processing function
            judge_process_dataset(
                dataset_path=dataset_path_test,
                error_path=error_path,
                max_workers=max_workers,
                start_index=0
            )

            print(f"{dataset_path_test} completed")

    print("Step 3 completed!")

    print("\nAll processing steps completed!")


if __name__ == "__main__":
    main()
