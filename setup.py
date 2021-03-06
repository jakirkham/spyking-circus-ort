from __future__ import print_function
from setuptools import setup, find_packages
import os
from os.path import join as pjoin
import sys, subprocess, re


requires = ['numpy', 'tqdm']

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if sys.version_info < (2, 7):
    raise RuntimeError('Only Python versions >= 2.7 are supported')

curdir = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(curdir, 'circusort/__init__.py')
with open(filename, 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


def _package_tree(pkgroot):
    path = os.path.dirname(__file__)
    subdirs = [os.path.relpath(i[0], path).replace(os.path.sep, '.')
               for i in os.walk(os.path.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return subdirs

setup(name='circusort',
      version=version,
      description='Fast spike sorting by template matching',
      long_description=read('README.rst'),
      url='http://spyking-circus.rtfd.org',
      author='Pierre Yger, Baptiste Lefebvre and Olivier Marre',
      author_email='pierre.yger@inserm.fr',
      license='License :: OSI Approved :: UPMC CNRS INSERM Logiciel Libre License, version 2.1 (CeCILL-2.1)',
      keywords="spike sorting template matching tetrodes extracellular",
      packages=_package_tree('circusort'),
      setup_requires=['setuptools>0.18'],
      install_requires=requires,
      use_2to3=True,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: Other/Proprietary License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      zip_safe=False)