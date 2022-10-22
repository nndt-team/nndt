import os
import os.path
import requests
import re
import urllib
import urllib.error
import urllib.request
import gdown
import hashlib
import py7zr


def _download_from_url(url: str, path_to: str, chunk_size: int = 32768, extension: str = '7z') -> str:
    path_to = path_to + 'temp.' + extension
    r = requests.get(url, stream=True)
    with open(path_to, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

    return path_to


def _download_from_google(url: str, path_to: str = './', extension: str = '7z') -> str:
    path_to = path_to + 'temp.' + extension

    def get_id(url):
        parts = urllib.parse.urlparse(url)
        match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
        return match.group("id")

    id = get_id(url)
    google_link = "https://drive.google.com/uc?export=download&id=" + id

    gdown.download(google_link, path_to, quiet=False)
    return path_to


def _extract_7z_file(from_path: str, to_path: str, delete_archive: bool = False) -> None:
    with py7zr.SevenZipFile(from_path, mode='r') as z:
        z.extractall(path=to_path)

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
