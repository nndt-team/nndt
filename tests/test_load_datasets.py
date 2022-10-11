from nndt.datasets import load_acdc

load_acdc()
#%%
import gdown
import zipfile
import urllib
import re
import datasets
import requests

url = datasets.ACDC_5().url


def get_id(url):
    parts = urllib.parse.urlparse(url)
    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    return match.group("id")

id = get_id(url)


url_to_download = "https://drive.google.com/uc?export=download&id=" + id
#%%
print(1)
gdown.download(url_to_download, './a.zip', quiet=True)
print(2)
#%%

def _extract_zip(from_path: str, to_path: str) -> None:
    print('Extracting data')
    with zipfile.ZipFile(
            from_path, "r", compression=zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)
#%%
_extract_zip('./a.zip', './a/')
#%%
import requests, zipfile, io
url = "https://www.dropbox.com/s/m4mz82s5cyotcva/ACDC_5.zip?raw=1"
r = requests.get(url, stream=True)
r
#%%
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./test_dir/")
#%%
