import os
import requests
import zipfile
from tqdm import tqdm
from huggingface_hub import snapshot_download

def download_file_with_progress(url, destination):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def get_iam_lines():
    print("\n--- 1. Downloading IAM-lines from Hugging Face ---")
    iam_dir = "data/raw/IAM"
    os.makedirs(iam_dir, exist_ok=True)
    
    # Use the native Python API instead of the CLI to avoid PATH errors
    snapshot_download(
        repo_id="Teklia/IAM-line", 
        repo_type="dataset", 
        local_dir=iam_dir
    )
    print(f"IAM-lines downloaded to {iam_dir}/")

def get_figshare_math_datasets(article_id="30341779", base_dir="data/raw"):
    print("\n--- 2. Fetching CROHME & HME100K from Figshare ---")
    api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()

    os.makedirs(base_dir, exist_ok=True)

    for f in files:
        file_name = f['name']
        download_url = f['download_url']
        save_path = os.path.join(base_dir, file_name)
        
        # Route the extraction based on the filename
        if "crohme" in file_name.lower() or "chrome" in file_name.lower():
            extract_dir = os.path.join(base_dir, "CHROME")
        elif "hme100k" in file_name.lower():
            extract_dir = os.path.join(base_dir, "HME100K")
        else:
            extract_dir = os.path.join(base_dir, file_name.replace('.zip', ''))

        os.makedirs(extract_dir, exist_ok=True)
        print(f"\nDownloading: {file_name}")
        download_file_with_progress(download_url, save_path)
        
        if file_name.endswith('.zip'):
            print(f"Extracting {file_name} into {extract_dir}/ ...")
            try:
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                # Clean up the zip file immediately
                os.remove(save_path)
                print(f"Deleted {file_name} archive to save space.")
            except zipfile.BadZipFile:
                print(f"Error: {file_name} is not a valid zip file.")

if __name__ == "__main__":
    # Ensure our raw data directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Execute the downloads
    get_iam_lines()
    get_figshare_math_datasets()
    
    print("\nAll datasets have been successfully downloaded and unzipped into their respective folders!")