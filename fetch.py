import urllib.request
import zipfile
import os

def download_cricsheet_ipl():
    url = "https://cricsheet.org/downloads/ipl_csv2.zip"
    os.makedirs("data/raw", exist_ok=True)
    zip_path = "data/ipl_csv2.zip"
    
    print("Downloading IPL data from Cricsheet...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw")
        
    os.remove(zip_path)
    print("Download and extraction complete. Files are in 'data/raw/'")

if __name__ == "__main__":
    download_cricsheet_ipl()