from nndt.datasets.dataset import Dataset


class MICCAI2015(Dataset):
    """
    MICCAI2015 includes 50 CT scans and ground truth segmentation of 13 abdominal organs.

    NNDT DOES NOT DOWNLOAD THE ORIGINAL DATA!
    We preprocess some subsets of models to create the toy examples for testing NNDT capabilities.

    Available out-of-the-box model subsets:
     - `left_adrenal_gland_10` includes 10 models of the adrenal glands
     - `stomach_10` includes 10 models of the stomach.

    Source: https://www.synapse.org/#!Synapse:syn3193805/wiki/217789
    DOI: https://doi.org/10.7303/syn3193805
    """

    def __init__(self, name="left_adrenal_gland_10", to_path=None):
        super().__init__(name=name, to_path=to_path)

        self._dict = {
            "left_adrenal_gland_10": [
                [
                    "https://drive.google.com/file/d/1LFJs5tW-acZOXTVVHh-WpWRNkAgtKaDd/view?usp=sharing",
                    "https://www.dropbox.com/s/xyarraopmj069de/left_adrenal_gland.7z?raw=1",
                ],
                "03c5f5c7e33a57ed71f497725d6925db",
            ],
            "stomach_10": [
                [
                    "https://drive.google.com/file/d/189tTsVX89qUijClkPkZfBiX5hRMupW03/view?usp=sharing",
                    "https://www.dropbox.com/s/uzhurjucqqnansh/stomach.7z?raw=1",
                ],
                "3a7a817748c43194fa75b1e5ab798d56",
            ],
            "wrong_url_test": [
                [
                    "https://drive.google.com/file",
                    "a",
                    "b",
                    "https://a.com",
                    "https://www.dropbox.com/s/m",
                ],
                "03c5f5c7e33a57ed71f497725d6925db",
            ],
            "wrong_hash_test": [
                [
                    "https://drive.google.com/file/d/1UzC2WPkjMQSxzI5sj1rMT47URuZbQhYb/view?usp=sharing",
                    "https://www.dropbox.com/s/6fomqxbjs0iu79m/ACDC_5.7z?raw=1",
                ],
                "34d007546353673899c73337d38c9c12  ",
            ],
            "dropbox_test": [
                [
                    "https://www.dropbox.com/s/xyarraopmj069de/left_adrenal_gland.7z?raw=1"
                ],
                "03c5f5c7e33a57ed71f497725d6925db",
            ],
        }

        if name in self._dict.keys():
            self.name = "MICCAI2015"
            self.to_path = to_path
            self.urls, self.hash = self._dict[name]

        else:
            raise ValueError(
                f'name must be in {[key for key in self._dict if "_test" not in key]}'
            )
