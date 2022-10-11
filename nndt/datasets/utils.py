import os
import os.path
import requests
import io
import re
import urllib
import urllib.error
import urllib.request
import zipfile
import gdown

from acdc import ACDC_5, ACDC_TEST, ACDC_TEST2


# TODO: create datasets.py and move func load there
def load(dataset: str, to_path: str = None) -> None:  # TODO: create class for datasets and change -> None to -> Dataset
    # TODO add assert for dataset that it is in list_of_datasets from __init__
    if dataset == 'ACDC_5':
        urls = ACDC_5().urls  # TODO: make it universal
    elif dataset == 'test':
        urls = ACDC_TEST().urls
    elif dataset == 'test2':
        urls = ACDC_TEST2().urls
        dataset='ACDC_5'

    if to_path is None:
        to_path = f'./{dataset}'
    complete = False

    for idx in range(len(urls)):
        url = urls[idx]
        if 'drive.google' in url:
            try:
                z = _download_from_google(url)
                _extract_zip_file(z, to_path)
            except Exception as e:  # TODO: rewrite exceptions
                print('Something went wrong:')
                print(str(e))
                print(f'Trying to use mirror {idx +1}/{len(urls)}')
                continue
        else:
            try:
                r = _download_from_url(url)
                _extract_zip_from_memory(r, to_path)
            except Exception as e:  # TODO: rewrite exceptions
                print('Something went wrong:')
                print(str(e))
                print(f'Trying to use mirror {idx + 1}/{len(urls)}')
                continue
        complete = True
        print('Loading complete')
        break
    if not complete:
        raise ConnectionError("Looks like you can't reach any mirror, "
              "please report this issue to: https://github.com/KonstantinUshenin/nndt")


def _extract_zip_from_memory(response: requests.models.Response, to_path: str) -> None:
    # Extracts zip archive received from response. Is used after _download_from_url func
    print('Extracting files')
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(to_path)


def _download_from_url(url: str) -> requests.models.Response:  # TODO: add progress bar
    # Downloads a zip archive from url and stores it im computer memory
    print('Response sent')
    response = requests.get(url, stream=True)
    print('Response received')
    return response


def _download_from_google(url: str, path_to: str = './', verbose=True) -> str:
    # Downloads a file from Google Drive and saves it to disk.
    path_to = path_to + 'temp.zip'

    def get_id(url):
        parts = urllib.parse.urlparse(url)
        match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
        return match.group("id")

    id = get_id(url)
    google_link = "https://drive.google.com/uc?export=download&id=" + id

    if verbose:
        print('Starting to download')
    gdown.download(google_link, path_to, quiet=not verbose)

    if verbose:
        print('Finished downloading')
    return path_to


def _extract_zip_file(from_path: str, to_path: str, delete_archive: bool = True) -> None:
    # Extracts zip file received from Google Drive and removes it
    print('Extracting data')
    with zipfile.ZipFile(from_path, "r", compression=zipfile.ZIP_STORED) as zip:
        zip.extractall(to_path)
    if delete_archive:
        os.remove(from_path)
