import os
import warnings
from pathlib import Path

from nndt import datasets
from nndt.datasets.utils import (
    _check_md5,
    _download_from_google,
    _download_from_url,
    _extract_7z_file,
)


class Dataset:
    def __init__(self, name=None, to_path=None):
        self.name = name
        self.to_path = to_path
        self.hash = None
        self.urls = None
        self._dict = None

    def dataset_list(self):
        """
        Get available subsets of the models

        :return: list of the available datasets for the download
        """
        return [key for key in self._dict if "_test" not in key]

    def load(self) -> str:
        """
        Load the dataset and return the path to its location.

        :return: path to dataset location
        """

        if self.to_path is None:
            self.to_path = f"./.datasets/{self.name}/"

        Path(self.to_path).mkdir(parents=True, exist_ok=True)
        complete = False

        for url in self.urls:
            if "drive.google" in url:
                try:
                    z = _download_from_google(url, self.to_path)
                    assert _check_md5(z, self.hash)
                    _extract_7z_file(z, self.to_path)
                except Exception as e:
                    warnings.warn(str(e))
                    continue
            else:
                try:
                    print("Downloading...")
                    z = _download_from_url(url, self.to_path)
                    assert _check_md5(z, self.hash)
                    _extract_7z_file(z, self.to_path)
                except Exception as e:
                    warnings.warn(str(e))
                    continue
            complete = True
            os.remove(z)
            print("Loading complete")
            break
        if not complete:
            raise ConnectionError(
                "Looks like you can't reach any mirror, "
                f"please report this issue to: {datasets.source_url}"
            )

        return self.to_path
