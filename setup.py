from setuptools import setup

setup(
	name='tensorplane',
	version='0.0.1',
	author='Jonathan Hanna',
	packages=['tensorplane'],
	license='LICENSE.txt',
	url='git@github.com:JonathanHanna/tensorplane.git',
	description='A modular and expressive interface for managing data in Python',
	long_description=open('README.md').read(),
	install_requires=[
        "numpy >= 1.15.0",
	],
)
