from space.abstracts import AbstractSource, ExtendedNodeMixin


class MeshSource(AbstractSource, ExtendedNodeMixin):

    def __init__(self, name, filepath, parent=None):
        super(MeshSource, self).__init__(filepath)
        self.name = name
        self.parent = parent


class SDTSource(AbstractSource, ExtendedNodeMixin):

    def __init__(self, name, filepath, parent=None):
        super(SDTSource, self).__init__(filepath)
        self.name = name
        self.parent = parent
