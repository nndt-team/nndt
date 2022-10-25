import io
import os
import re

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def find_requirements(*file_path):
    requirements = read(*file_path).split("\n")
    return [req for req in requirements if req]


version = find_version("nndt", "__init__.py")
requirements = find_requirements("requirements.txt")
readme = open("README.md").read()

setup(
    name="NNDT",
    version=version,
    author="Konstantin Ushenin",
    author_email="k.ushenin@gmail.com",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Neural network for human digital twin",
    long_description=readme,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/KonstantinUshenin/nndt",
    },
    keywords=["pinn implicit-geometry jax machine-learning"],
    install_requires=requirements,
)
