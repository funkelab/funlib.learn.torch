from setuptools import setup
from setuptools.extension import Extension
from distutils.sysconfig import get_python_inc, get_config_var
import os

import numpy as np

losses_impl_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'funlib',
    'learn',
    'tensorflow',
    'losses',
    'impl')

include_dirs = [
    losses_impl_dir,
    os.path.dirname(get_python_inc()),
    get_python_inc(),
    np.get_include()
]

library_dirs = [
    losses_impl_dir,
    get_config_var("LIBDIR")
]


setup(
        name='funlib.learn.torch',
        version='0.1',
        url='https://github.com/funkelab/funlib.learn.torch',
        author='Funkelab',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        setup_requires=[
            'cython'
        ],
        packages=[
            'funlib.learn.torch',
            'funlib.learn.torch.models',
            'funlib.learn.torch.misc',
            'funlib.learn.torch.losses',
            'funlib.learn.torch.losses.impl',
            'funlib.learn.torch.ext',
        ],
        ext_modules=[
            Extension(
                'funlib.learn.torch.losses.impl.wrappers',
                sources=[
                    'funlib/learn/torch/losses/impl/wrappers.pyx',
                    'funlib/learn/torch/losses/impl/um_loss.cpp',
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                extra_link_args=['-std=c++11'],
                extra_compile_args=['-O3', '-std=c++11'],
                language='c++')
        ]
)
