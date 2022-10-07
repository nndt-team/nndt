from abc import abstractmethod

from space2 import FileSource


class AbstractLoader():

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def unload_data(self):
        pass

    @abstractmethod
    def is_load(self) -> bool:
        pass


class EmptyLoader(AbstractLoader):

    def __init__(self, filesource: FileSource):
        self.filesource = filesource
        self.is_load = False

    def load_data(self):
        self.is_load = True

    def unload_data(self):
        self.is_load = False

    def is_load(self) -> bool:
        return self.is_load


DICT_LOADERTYPE_CLASS = {'txt': EmptyLoader,
                    'sdt': EmptyLoader,
                    'mesh_obj': EmptyLoader,
                    'undefined': EmptyLoader}
DICT_CLASS_LOADERTYPE = {(v, k) for k, v in DICT_LOADERTYPE_CLASS.items()}