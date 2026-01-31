"""
PIANO: Probabilistic Inference Autoencoder Networks for multi-Omics
Copyright (C) 2025 Ning Wang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from setuptools import setup, find_packages

extra_reqs = {
    'rapids': [
        'rapids-singlecell[rapids12]==0.13.1'
    ],
    'scvi-tools': [
        'jax[cuda12]',
        'scvi-tools',
        'scib-metrics==0.5.3'
    ],
    'torch': [
        'torchvision',
        'torchaudio'
    ],
    'misc': [
        'igraph',
        'leidenalg',
        'memory_profiler',
        'seaborn',
        'joblib',
        'jupyterlab',
        'pot'
    ],
}
extra_reqs["all"] = sorted({pkg for group in extra_reqs.values() for pkg in group})
for _ in extra_reqs:
    extra_reqs[_].insert(0, 'torch>=2.2,<2.8')

with open('README.md', mode='r') as readme:
    long_description=readme.read()

setup(
    name='PIANO',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Machine learning imports
        'fast-array-utils',
        'matplotlib',
        'numpy>=2',
        'pandas',
        'scipy',
        'scikit-misc',
        # Single cell imports
        'anndata',
        'scanpy',
        # PyTorch imports
        'torch>=2.2,<2.8',
        'pyro-ppl'
    ],
    extras_require=extra_reqs,
    author='Ning Wang',
    description='PIANO - Probabilistic Inference Autoencoder Networks for multi-Omics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ningwang1729/piano',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    license='GPLv3',
)
