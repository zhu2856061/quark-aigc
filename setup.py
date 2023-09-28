# -*- coding: utf-8 -*-
# @Time   : 2021/10/27 15:47
# @Author : zip
# @Moto   : Knowledge comes from decomposition
import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'omegaconf==2.0.6',
    'pytorch-lightning>=2.0.4',
    'torchdata>=0.6.1',
    'torchmetrics>=0.11.4',
    'torch>=2.0.0',
    'loguru>=0.7.0',
    'cython>=0.29.32',
    'loguru>=0.7.0',
    'transformers>=4.32.1',
    'diffusers>=0.21.3',
    'invisible_watermark>=0.1.5',
    'accelerate>=0.23.0',
    'safetensors>=0.3.1'
]


setuptools.setup(
    name="quarkaigc",
    version="0.0.1",
    author="merlin",
    description="easy-to-use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="None",
    packages=setuptools.find_packages(exclude=["experiment", "tools", "test"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=REQUIRED_PACKAGES,
    entry_points={},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="Apache-2.0",
    keywords=['train framework', 'deep learning'],
)

# python setup.py bdist_wheel
