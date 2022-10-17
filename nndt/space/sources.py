from nndt.space.abstracts import FileSource, ExtendedNodeMixin, AbstractSource


class MeshSource(FileSource, ExtendedNodeMixin):

    def __init__(self, name, filepath, parent=None):
        super(MeshSource, self).__init__(filepath)
        self.name = name
        self.parent = parent


class SDTSource(FileSource, ExtendedNodeMixin):

    def __init__(self, name, filepath, parent=None):
        super(SDTSource, self).__init__(filepath)
        self.name = name
        self.parent = parent


class SDFPKLSource(FileSource, ExtendedNodeMixin):

    def __init__(self, name, filepath, parent=None):
        super(SDFPKLSource, self).__init__(filepath)
        self.name = name
        self.parent = parent


class SphereSDFSource(AbstractSource, ExtendedNodeMixin):

    def __init__(self, name, center=(0., 0., 0.), radius=1., parent=None):
        super(SphereSDFSource, self).__init__()
        self.name = name
        self.parent = parent

        self.center = center
        self.radius = radius
