import os
from huggingface_hub import snapshot_download

def download_data(repo_id="Neatherblok/SnowySidewalkDetection-SyntheticEval",
                  cache_dir="./data/synthetic_eval"):
    """
    Downloads the dataset from the specified Hugging Face repository.

    Args:
        repo_id (str): The Hugging Face repository ID to download from.
        cache_dir (str): The directory to store the downloaded data.

    Returns:
        dict: Paths to the downloaded dataset folders.
    """
    print(f"Downloading dataset from Hugging Face repository: {repo_id}")
    data_path = snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=cache_dir)
    print(f"Dataset downloaded to: {data_path}")

    # Identify folders
    folders = {
        "DALL-E3": None,
        "Grok2": None,
        "Imagen3": None,
        "RealWorld": None
    }

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            if folder in folders:
                folders[folder] = folder_path

    return folders

if __name__ == "__main__":
    # Ensure the directory for downloaded data exists
    os.makedirs("./data/synthetic_eval", exist_ok=True)

    # Download the dataset
    folders = download_data()

    # Print the location of the dataset folders
    print("Dataset Folders:")
    for folder_name, folder_path in folders.items():
        if folder_path:
            print(f"{folder_name}: {folder_path}")
        else:
            print(f"{folder_name}: Not found")
