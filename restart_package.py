"""restart package

This script allows the user to uninstall people emotions package,
remove dist, build and people_emotions.egg-info folders and
re-install the package.

Usage:
    python restart_package.py

Author:
    Ahmed Haj Yahme (hajyahmedahmed@gmail.com)
"""
import os
import shutil
os.system('pip uninstall people_emotions')
shutil.rmtree('./dist', ignore_errors=True)
shutil.rmtree('./build', ignore_errors=True)
shutil.rmtree('./people_emotions.egg-info', ignore_errors=True)
os.system('python setup.py sdist bdist_wheel')
os.system('pip install dist/people_emotions-0.1-py3-none-any.whl')
