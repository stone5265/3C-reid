import numpy as np
from setuptools import setup, find_packages
from Cython.Build import cythonize


# if __name__ == '__main__':
#     import sys
#     sys.argv = ['setup.py', 'clean', 'build_ext', '-i']

setup(
    install_requires=[
        'torch', 'torchvision', 'Pillow',
        'faiss_gpu', 'scikit-learn', 'scikit-learn-intelex'
    ],
    packages=find_packages(),
    ext_modules=cythonize([
        'CCC/HDC/_HDC_utils.pyx',
        'CCC/utils/_rerank.pyx'
    ], compiler_directives={'language_level': '3'}),
    include_dirs=[np.get_include()]
)
