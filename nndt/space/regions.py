from colorama import Fore

from nndt.space.abstracts import AbstractRegion, ExtendedNodeMixin


class Space(AbstractRegion, ExtendedNodeMixin):

    def __init__(self, name, parent=None):
        super(Space, self).__init__()
        self.name = name
        self.parent = parent
        self._print_color = Fore.RED


class Group(AbstractRegion, ExtendedNodeMixin):

    def __init__(self, name, parent=None):
        super(Group, self).__init__()
        self.name = name
        self.parent = parent
        self._print_color = Fore.RED


class Object(AbstractRegion, ExtendedNodeMixin):

    def __init__(self, name, parent=None):
        super(Object, self).__init__()
        self.name = name
        self.parent = parent
        self._print_color = Fore.BLUE
