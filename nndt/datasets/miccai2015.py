from nndt.datasets.dataset import dataset


class MICCAI2015(dataset):
    def __init__(self, name='left_adrenal_gland_10', to_path=None):
        super().__init__()

        __dict = {
            'left_adrenal_gland_10': [
                ['https://drive.google.com/file/d/1LFJs5tW-acZOXTVVHh-WpWRNkAgtKaDd/view?usp=sharing',
                 'https://www.dropbox.com/s/xyarraopmj069de/left_adrenal_gland.7z?raw=1'],
                '03c5f5c7e33a57ed71f497725d6925db'

            ],
            'stomach_10': [
                ['https://drive.google.com/file/d/189tTsVX89qUijClkPkZfBiX5hRMupW03/view?usp=sharing',
                 'https://www.dropbox.com/s/uzhurjucqqnansh/stomach.7z?raw=1'],
                '3a7a817748c43194fa75b1e5ab798d56'
            ],
            'wrong_url_test': [
                ['https://drive.google.com/file',
                 'a',
                 'b',
                 'https://a.com',
                 'https://www.dropbox.com/s/m'],
                '03c5f5c7e33a57ed71f497725d6925db'

            ],
            'wrong_hash_test': [
                ['https://drive.google.com/file/d/1UzC2WPkjMQSxzI5sj1rMT47URuZbQhYb/view?usp=sharing',
                 'https://www.dropbox.com/s/6fomqxbjs0iu79m/ACDC_5.7z?raw=1'],
                '34d007546353673899c73337d38c9c12  '],
            'dropbox_test': [
                ['https://www.dropbox.com/s/xyarraopmj069de/left_adrenal_gland.7z?raw=1'],
                '03c5f5c7e33a57ed71f497725d6925db'
            ]
        }

        if name in __dict.keys():
            self.name = 'MICCAI2015'
            self.to_path = to_path
            self.urls, self.hash = __dict[name]

        else:
            raise ValueError(f'name must be in {[key for key in __dict if "test" not in key]}')

    def load(self):
        super(MICCAI2015, self).load()
