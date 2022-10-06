import os
import requests
from amos import AMOS_5
import urllib
import itertools
import re

#%%
dataset = AMOS_5()
url = dataset.url



def get_id(url):
    parts = urllib.parse.urlparse(url)
    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    return match.group("id")

#%%
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = url
    destination = './amos.7z'
    download_file_from_google_drive(file_id, destination)






# #%%
# def download_file_from_google_drive(file_id: str,
#                                     root: str,
#                                     filename: str = None,
#                                     md5: str = None):
#     root = os.path.expanduser(root)
#     if not filename:
#         filename = file_id
#     fpath = os.path.join(root, filename)
#
#     os.makedirs(root, exist_ok=True)
#
#     url = "https://drive.google.com/uc"
#     params = dict(id=file_id, export="download")
#     with requests.Session() as session:
#         response = session.get(url, params=params, stream=True)
#     return response
#
# resp = download_file_from_google_drive(get_id(url), root='.')
# #%%
# def _extract_gdrive_api_response(response, chunk_size: int = 32 * 1024) -> [bytes,[bytes]]:
#     content = response.iter_content(chunk_size)
#     first_chunk = None
#     # filter out keep-alive new chunks
#     while not first_chunk:
#         first_chunk = next(content)
#     content = itertools.chain([first_chunk], content)
#
#     try:
#         match = re.search("<title>Google Drive - (?P<api_response>.+?)</title>", first_chunk.decode())
#         api_response = match["api_response"] if match is not None else None
#     except UnicodeDecodeError:
#         api_response = None
#     return api_response, content
# #%%
# r, content = _extract_gdrive_api_response(resp)
# #%%
# r


#%%
    #     for key, value in response.cookies.items():
    #         if key.startswith("download_warning"):
    #             token = value
    #             break
    #     else:
    #         api_response, content = _extract_gdrive_api_response(response)
    #         token = "t" if api_response == "Virus scan warning" else None
    #
    #     if token is not None:
    #         response = session.get(url, params=dict(params, confirm=token), stream=True)
    #         api_response, content = _extract_gdrive_api_response(response)
    #
    #     if api_response == "Quota exceeded":
    #         raise RuntimeError(
    #             f"The daily quota of the file {filename} is exceeded and it "
    #             f"can't be downloaded. This is a limitation of Google Drive "
    #             f"and can only be overcome by trying again later."
    #         )
    #
    #     _save_response_content(content, fpath)
    #
    # # In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB and contain only text
    # if os.stat(fpath).st_size < 10 * 1024:
    #     with contextlib.suppress(UnicodeDecodeError), open(fpath) as fh:
    #         text = fh.read()
    #         # Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604
    #         if re.search(r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)", text):
    #             warnings.warn(
    #                 f"We detected some HTML elements in the downloaded file. "
    #                 f"This most likely means that the download triggered an unhandled API response by GDrive. "
    #                 f"Please report this to torchvision at https://github.com/pytorch/vision/issues including "
    #                 f"the response:\n\n{text}"
    #             )
    #
    # if md5 and not check_md5(fpath, md5):
    #     raise RuntimeError(
    #         f"The MD5 checksum of the download file {fpath} does not match the one on record."
    #         f"Please delete the file and try again. "
    #         f"If the issue persists, please report this to torchvision at https://github.com/pytorch/vision/issues."
    #     )
    #
    #


# #%%
# def download_and_extract_archive(
#         url: str,
#         download_root: str,
#         extract_root: str = None,
#         filename: str = None,
#         md5: str = None,
#         remove_finished: bool = False,
# ) -> None:
#     download_root = os.path.expanduser(download_root)
#     if extract_root is None:
#         extract_root = download_root
#     if not filename:
#         filename = os.path.basename(url)
#
#     download_url(url, download_root, filename, md5)
#
#     archive = os.path.join(download_root, filename)
#     print(f"Extracting {archive} to {extract_root}")
#     _extract_zip(archive, extract_root)

