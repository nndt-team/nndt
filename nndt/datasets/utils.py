import os
import os.path
import pathlib
import re
import sys
import tarfile
import urllib
import urllib.error
import urllib.request

import zipfile


def load(name, download=True):
    pass


def _extract_zip(from_path: str, to_path: str) -> None:
    with zipfile.ZipFile(
            from_path, "r", compression=zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)


def download_and_extract_archive(
        url: str,
        download_root: str,
        extract_root: str = None,
        filename: str = None,
        md5: str = None,
        remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
    _extract_zip(archive, extract_root)


def download_url():
    pass
#    (
#         url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None, max_redirect_hops: int = 3
# ) -> None:
#     """Download a file from a url and place it in root.
#     Args:
#         url (str): URL to download file from
#         root (str): Directory to place downloaded file in
#         filename (str, optional): Name to save the file under. If None, use the basename of the URL
#         md5 (str, optional): MD5 checksum of the download. If None, do not check
#         max_redirect_hops (int, optional): Maximum number of redirect hops allowed
#     """
#     root = os.path.expanduser(root)
#     if not filename:
#         filename = os.path.basename(url)
#     fpath = os.path.join(root, filename)
#
#     os.makedirs(root, exist_ok=True)
#
#     # check if file is already present locally
#     if check_integrity(fpath, md5):
#         print("Using downloaded and verified file: " + fpath)
#         return
#
#     if _is_remote_location_available():
#         _download_file_from_remote_location(fpath, url)
#     else:
#         # expand redirect chain if needed
#         url = _get_redirect_url(url, max_hops=max_redirect_hops)
#
#         # check if file is located on Google Drive
#         file_id = _get_google_drive_file_id(url)
#         if file_id is not None:
#             return download_file_from_google_drive(file_id, root, filename, md5)
#
#         # download the file
#         try:
#             print("Downloading " + url + " to " + fpath)
#             _urlretrieve(url, fpath)
#         except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
#             if url[:5] == "https":
#                 url = url.replace("https:", "http:")
#                 print("Failed download. Trying https -> http instead. Downloading " + url + " to " + fpath)
#                 _urlretrieve(url, fpath)
#             else:
#                 raise e
#
#     # check integrity of downloaded file
#     if not check_integrity(fpath, md5):
#         raise RuntimeError("File not found or corrupted.")
