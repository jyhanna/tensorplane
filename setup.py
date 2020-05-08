import os
import re
from setuptools import setup, find_packages


def parse_readme():
	with open('README.md', encoding='utf-8') as f:
		txt = re.sub(re.compile('<.*?>'), '', f.read())
		txt = txt[re.search(r"[a-zA-Z#]", txt).start():]
		return txt


setup(
	name='tensorplane',
	version='0.0.1',
	author='Jonathan Hanna',
	author_email='jonathanyussefhanna@gmail.com',
	packages=find_packages(),
	license='LICENSE.txt',
	url='https://github.com/JonathanHanna/tensorplane',
	description='A modular and expressive interface for managing data in Python',
	long_description=parse_readme(),
	install_requires=[
		"numpy >= 1.15.0",
	],
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Developers',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'License :: OSI Approved :: BSD License',
		'Programming Language :: Python :: 3'
	]
)
