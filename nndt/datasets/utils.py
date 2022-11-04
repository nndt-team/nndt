import hashlib
import os
import os.path
import re
import urllib
import urllib.error
import urllib.request

import gdown
import py7zr
import requests
from tqdm import tqdm


def _download_from_url(
    url: str, path_to: str, chunk_size: int = 32768, extension: str = "7z"
) -> str:
    path_to = path_to + "temp." + extension
    response = requests.get(url, stream=True)
    total_length = response.headers.get("content-length")
    total_length = int(total_length) if total_length.isdigit() else None

    with open(path_to, "wb") as fd:
        if total_length is not None:
            tmp_iter = tqdm(
                response.iter_content(chunk_size=chunk_size),
                total=int(total_length / chunk_size) + 1,
            )
        else:
            tmp_iter = tqdm(response.iter_content(chunk_size=chunk_size))

        for chunk in tmp_iter:
            fd.write(chunk)
        fd.flush()

    return path_to


def _download_from_google(url: str, path_to: str = "./", extension: str = "7z") -> str:
    path_to = path_to + "temp." + extension

    def get_id(url):
        parts = urllib.parse.urlparse(url)
        match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
        return match.group("id")

    id = get_id(url)
    google_link = "https://drive.google.com/uc?export=download&id=" + id

    gdown.download(google_link, path_to, quiet=False)
    return path_to


def _extract_7z_file(
    from_path: str, to_path: str, delete_archive: bool = False
) -> None:
    with py7zr.SevenZipFile(from_path, mode="r") as z:
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
    with open(file, "rb") as f:
        data = f.read()
        md5 = hashlib.md5(data).hexdigest()

        if orig_hash == md5:
            return True
        else:
            return False
