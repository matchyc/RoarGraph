from glob import glob
from setuptools import setup, Extension
import numpy as np
import pybind11
from setuptools.command.build_ext import build_ext
import setuptools
import os
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext

include_dirs = [
    # Path to pybind11 headers
    pybind11.get_include(),
    np.get_include()
]

include_dirs.extend(['../include/', '../include/efanna2e', '../thirdparty/robin-map/include'])

module = Pybind11Extension('RoarGraph',
             sources=['index_bindings.cpp',
                      '../src/index.cpp', '../src/index_bipartite.cpp'
                      ],
             include_dirs=include_dirs,
            #  libraries=['mysteryann'],
             libraries=['faiss_avx2'],
             language='c++')
# module = [Pybind11Extension(
#         "python_example",
#         sorted(glob("../src/*.cpp")),
#         "index_bindings.cpp",
#         )]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """

    # if has_flag(compiler, '-std=c++14'):
        # return '-std=c++14'
    # elif has_flag(compiler, '-std=c++11'):
        # return '-std=c++11'
    if has_flag(compiler, '-std=c++20'):
        return '-std=c++20'
    elif has_flag(compiler, '-std=c++17'):
        return '-std=c++17'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'unix': ['-O3', '-fopenmp',
                 '-DNDEBUG', 
                #  '-DDEBUG', '-g', # uncomment this line if you want to debug
                 '-march=native', '-mtune=native', '-ftree-vectorize','-Wall', '-DINFO', '-mavx2', 
                #  '-mavx512f', '-mavx512cd', '-mavx512dq', '-mavx512bw', '-mavx512vl' # uncomment this line if you have avx512
                 ],  # , '-w'
    }
    
    # c_opts['unix'].append('-march=native')

    link_opts = {
        'unix': [],
    }

    c_opts['unix'].append("-fopenmp")
    link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        # if ct == 'unix':
            # opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
        opts.append(cpp_flag(self.compiler))
            # if has_flag(self.compiler, '-fvisibility=hidden'):
                # opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))

        build_ext.build_extensions(self)

setup(name='RoarGraph',
    version='1.0',
    description='Python bindings for RoarGraph class',
    ext_modules=[module],
    cmdclass={'build_ext': BuildExt},
    install_requires=['numpy', 'pybind11']
    )