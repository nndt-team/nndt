import os
import os.path
import pathlib
import re
import sys
import tarfile
import urllib
import urllib.error
import urllib.request
import requests
import zipfile
import shutil


from acdc import ACDC_5

path = 'file.7z'

def _extract_zip(from_path: str, to_path: str) -> None:
    print('Extracting data')
    with zipfile.ZipFile(
            from_path, "r", compression=zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)

def load_acdc(download=True):

    if not os.path.exists(path):
        download = True

    url = ACDC_5().url
    if 'google' in url:
        get_file_from_google(url)

    _extract_zip('file.7z', './')
    #os.remove('file.7z')
    print('Done')


def get_file_from_google(drive_url):
    def get_id(url):
        parts = urllib.parse.urlparse(url)
        match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
        return match.group("id")

    google_url = 'https://drive.google.com/uc?export=download&id='
    id = get_id(drive_url)

    url_to_download = google_url+id

    print('Request sent')
    r = requests.get(url_to_download, allow_redirects=True)
    print('Dataset received')

    with open('./file', 'wb') as fl:
        fl.write(r.content)
