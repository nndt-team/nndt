from nndt.datasets.dataset import Dataset


class ACDC(Dataset):
    """
    ACDC dataset includes 100 models of the left and right ventricles of the human heart.
    There are healthy patients and four types of pathologies.

    NNDT DOES NOT DOWNLOAD THE ORIGINAL DATA!
    We preprocess some subsets of models to create the toy examples for testing NNDT capabilities.

    Available out-of-the-box model subsets:
     - `ACDC_5` includes five models from ACDC. One healthy example and four pathologies.

    Source: https://www.creatis.insa-lyon.fr/Challenge/acdc/
    """

    def __init__(self, name="ACDC_5", to_path=None):
        super().__init__(name=name, to_path=to_path)

        self._dict = {
            "ACDC_5": [
                [
                    "https://drive.google.com/file/d/1UzC2WPkjMQSxzI5sj1rMT47URuZbQhYb/view?usp=sharing",
                    "https://www.dropbox.com/s/6fomqxbjs0iu79m/ACDC_5.7z?raw=1",
                ],
                "34d007546353673899c73337d38c9c12",
            ],
            "wrong_url_test": [
                [
                    "https://drive.google.com/file",
                    "a",
                    "b",
                    "https://a.com",
                    "https://www.dropbox.com/s/m",
                ],
                "34d007546353673899c73337d38c9c12",
            ],
            "wrong_hash_test": [
                [
                    "https://drive.google.com/file/d/1UzC2WPkjMQSxzI5sj1rMT47URuZbQhYb/view?usp=sharing",
                    "https://www.dropbox.com/s/6fomqxbjs0iu79m/ACDC_5.7z?raw=1",
                ],
                "34d007546353673899c73337d38c9c12  ",
            ],
            "dropbox_test": [
                ["https://www.dropbox.com/s/6fomqxbjs0iu79m/ACDC_5.7z?raw=1"],
                "34d007546353673899c73337d38c9c12",
            ],
        }

        if name in self._dict.keys():
            if "test" in name:
                self.name = "ACDC_5"
            else:
                self.name = name
            self.to_path = to_path
            self.urls, self.hash = self._dict[name]
        else:
            raise ValueError(
                f'name must be in {[key for key in self._dict if "_test" not in key]}'
            )
