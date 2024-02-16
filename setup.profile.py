from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

source_files = [
    'cython_src/dag.pyx',
    'cython_src/minisat/Options.cpp',
    'cython_src/minisat/SimpSolver.cpp',
    'cython_src/minisat/Solver.cpp',
    'cython_src/minisat/System.cpp']

setup(
    name = 'cython core',
    ext_modules = cythonize([Extension('dag', source_files,
                                       language = 'c++',
                                       extra_compile_args = ['-std=c++17', '-O2'],
                                       include_dirs=[numpy.get_include(), '/home/fs01/df394/miniconda3/include/eigen3/'],
                                       define_macros = [('CYTHON_TRACE', '1')],
                                      )],
                            compiler_directives={'language_level' : '3'},
                            annotate = True,
                           ),
    zip_safe = False,
)
