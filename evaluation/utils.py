"""

    These utility functions are used to download the official dataset from a public bucket
    and make it available to the rest of the code (see EvalRSRunner.py)

    You should not need to modify this script: if in doubt, ask the organizers first.

"""

from appdirs import *
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
from datetime import datetime


LFM_DATASET_PATH="https://evalrs.object.lga1.coreweave.com/evalrs_dataset_KDD_2023.zip"


def download_with_progress(url, destination):
    """
    Downloads a file with a progress bar

    :param url: url from which to download from
    :destination: file path for saving data
    """
    print(f"Downloading to {destination}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    with tqdm.wrapattr(open(destination, "wb"), "write",
                       miniters=1, desc=url.split('/')[-1],
                       total=int(response.headers.get('content-length', 0))) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


def decompress_zipfile(file, outfile):
    with zipfile.ZipFile(file , 'r') as zip_ref:
        zip_ref.extractall(outfile)


def get_cache_directory():
    """
    Returns the cache directory on the system
    """
    appname = "evalrs"
    appauthor = "evalrs"
    cache_dir = user_cache_dir(appname, appauthor)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir

