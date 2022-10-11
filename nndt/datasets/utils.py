import os
import os.path
import re
import urllib
import urllib.error
import urllib.request
import zipfile
import gdown

from acdc import ACDC_5

path = './acdc.zip'

def _extract_zip(from_path: str, to_path: str) -> None:
    print('Extracting data')
    with zipfile.ZipFile(
            from_path, "r", compression=zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)



def _download_from_google(download=True, verbose=True):
    url = ACDC_5().url

    def get_id(url):
        parts = urllib.parse.urlparse(url)
        match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
        return match.group("id")
    id = get_id(url)
    google_link = "https://drive.google.com/uc?export=download&id=" + id
    if verbose:
        print('Starting download')
    gdown.download(google_link, './a.zip', quiet=not verbose)
    if verbose:
        print('Finished downloading')


