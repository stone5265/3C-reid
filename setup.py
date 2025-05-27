import os.path as osp
import numpy as np
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize


def get_requirements(filename="requirements.txt"):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires
    
    
ext_modules = [
    Extension(
        "CCC.HDC._HDC_utils",
        ["CCC/HDC/_HDC_utils.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "CCC.utils._rerank",
        ["CCC/utils/_rerank.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()]
    )
]

if __name__ == "__main__":
    setup(
        name='3C-reid',
        version='0.1',
        author='Mingxiao Zheng',
        author_email='zheng_mx5265@foxmail.com',
        url='https://github.com/stone5265/3C-reid',
        install_requires=get_requirements(),
        packages=find_packages(),
        ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'})
    )
