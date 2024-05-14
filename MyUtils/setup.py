from setuptools import setup
import sys

if len(sys.argv) == 1:
    sys.argv.append('develop')

setup(
    name='MyUtils',
    version='0.1',
    py_modules=['MyUtils'],
)