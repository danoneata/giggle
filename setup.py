from setuptools import (
    setup,
    find_packages,
)


setup(
    name='giggle',
    description='A recommender system for jokes based on the Jester dataset.',
    version='1.0',
    author='Dan Oneata',
    author_email='dan.oneata@gmail.com',
    entry_points='''
        [console_scripts]
        giggle=giggle.cli:main
    ''',
    packages=find_packages(),
    license='MIT',
)
