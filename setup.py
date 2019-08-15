import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

extensions = [Extension('np_stream_median', [
                        'decision_tree/np_stream_median.pyx'],
                        extra_compile_args=["-Wno-cpp", "-Wno-unused-function",
                                            "-O2", "-march=native", '-stdlib=libc++', '-std=c++11'],
                        extra_link_args=[
                            "-O2", "-march=native", '-stdlib=libc++'],
                        language="c++",
                        include_dirs=[np.get_include()],
                        )]

setup(
    name='hello',
    ext_modules=cythonize(
        extensions
    )
)
