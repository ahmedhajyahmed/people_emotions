from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

setup_args = dict(
    name='people_emotions',
    version='0.1',
    description='package containing 3 models for detecting people emotion in a picture',
    long_description='package containing 3 models for detecting people emotion in a picture',
    classifiers=[
            'Programming Language :: Python :: 3.6.6',
            'Topic :: image Processing :: emotion detection',
          ],
    packages=find_packages(),
    author='Ahmed Haj Yahmed',
    author_email='hajyahmedahmed@gmail.com',
    keywords=['Face detection', 'Emotion detection', 'Image classification', 'Image segmentation'],
)

install_requires = requirements

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
