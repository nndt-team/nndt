import os
import os.path
import requests
import re
import urllib
import urllib.error
import urllib.request
import zipfile
import gdown
import hashlib
from pathlib import Path

from acdc import ACDC_5, _ACDC_TEST, _ACDC_TEST2


def load(dataset: str, to_path: str = None) -> None:
    if dataset == 'ACDC_5':
        urls = ACDC_5().urls
        hash = ACDC_5().hash

    elif dataset == 'test':
        urls = _ACDC_TEST().urls
        hash = _ACDC_TEST().hash

        dataset = 'ACDC_5'
    elif dataset == 'test2':
        urls = _ACDC_TEST2().urls
        hash = _ACDC_TEST2().hash

        dataset = 'ACDC_5'
    elif dataset == 'test_hash':
        urls = ACDC_5().urls
        hash = _ACDC_TEST().wrong_hash

        dataset = 'ACDC_5'
    else:
        raise ValueError("Please choose a dataset from datasets.list_of_datasets")

    if to_path is None:
        to_path = f'./.datasets/{dataset}/'

    Path(to_path).mkdir(parents=True, exist_ok=True)
    complete = False

    for idx in range(len(urls)):
        url = urls[idx]
        if 'drive.google' in url:
            try:
                z = _download_from_google(url, to_path)
                assert _check_md5(z, hash)
                _extract_zip_file(z, to_path)
            except Exception as e:
                continue
        else:
            try:
                print('Downloading...')
                z = _download_from_url(url, to_path)
                assert _check_md5(z, hash)
                _extract_zip_file(z, to_path)
            except Exception as e:
                continue
        complete = True
        os.remove(z)
        print('Loading complete')
        break
    if not complete:
        raise ConnectionError("Looks like you can't reach any mirror, "
                              "please report this issue to: https://github.com/KonstantinUshenin/nndt")


def _download_from_url(url: str, path_to: str, chunk_size: int = 32768) -> str:
    path_to = path_to + 'temp.zip'
    r = requests.get(url, stream=True)
    with open(path_to, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

    return path_to


def _download_from_google(url: str, path_to: str = './') -> str:
    path_to = path_to + 'temp.zip'

    def get_id(url):
        parts = urllib.parse.urlparse(url)
        match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
        return match.group("id")

    id = get_id(url)
    google_link = "https://drive.google.com/uc?export=download&id=" + id

    gdown.download(google_link, path_to, quiet=False)
    return path_to


def _extract_zip_file(from_path: str, to_path: str, delete_archive: bool = False) -> None:
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zip:
        zip.extractall(to_path)
    if delete_archive:
        os.remove(from_path)


def __create_md5(file: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _check_md5(file: str, orig_hash: str) -> bool:
    with open(file, 'rb') as f:
        data = f.read()
        md5 = hashlib.md5(data).hexdigest()

        if orig_hash == md5:
            return True
        else:
            return False
