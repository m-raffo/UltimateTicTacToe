from Cython.Distutils import build_ext

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension("gamestate", ["gamestate.pyx"], include_dirs=[numpy.get_include()],),
    Extension("mcts", ["mcts.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(extensions, annotate=True), cmdclass={"build_ext": build_ext}
)
# print(numpy.get_include())
# setup(
#     ext_modules=cythonize("mcts.pyx", "gamestate.pyx"),
#     include_dirs=[numpy.get_include()],
# )
