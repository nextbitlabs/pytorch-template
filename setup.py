import re
from codecs import open
from pathlib import Path

from setuptools import setup, find_packages

here = Path.resolve(Path(__file__).parent)


def read(*parts):
    with open(here.joinpath(*parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open(here / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pytorch_template',  # TODO: update
    version=find_version('pytorch_template', '__init__.py'),  # TODO: update
    description='PyTorch code and documentation template',  # TODO: update
    long_description=long_description,
    author='Nextbit AI Team',
    author_email='lgrementieri@nextbit.it',  # TODO: update
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Data scientists',  # TODO: update
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish
        'License :: Proprietary license',

        # Specify the Python versions you support here.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    keywords='AI deep learning pytorch template',  # TODO: update
    packages=find_packages(exclude=['build', 'data', 'dist', 'docs', 'tests']),
    python_requires='>=3.6',
    install_requires=[
        'numpy >= 1.13',
        'tensorboard>=2.0',
        'torch >= 1.3.1',
        'torchvision >= 0.2.1',
        'tqdm >= 4.23'
    ]
)
